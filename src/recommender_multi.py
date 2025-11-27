"""
Post recommender that supports multiple text encoders for easy comparison.

Pipeline mirrors the original:
- load CSVs
- embed text (batched) with a chosen encoder (BGE-M3 or SentenceTransformer variants)
- embed images with Dinov2
- learn a linear projection from image space to text space
- fuse embeddings, build user profiles, index with FAISS, and recommend
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
import pandas as pd
import torch
from FlagEmbedding import BGEM3FlagModel
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoImageProcessor, AutoModel

# Hardcoded data locations for clarity.
DATA_ROOT = Path("/home/alper/recsys/post-recsys/data")
POSTS_PATH = DATA_ROOT / "posts.csv"
LIKES_PATH = DATA_ROOT / "user_likes.csv"
IMAGES_DIR = DATA_ROOT / "images"


# ---------- Utilities ----------
def l2_normalize(array: np.ndarray) -> np.ndarray:
    """Normalize rows to unit length."""
    norms = np.linalg.norm(array, axis=1, keepdims=True) + 1e-12
    return array / norms


# ---------- Text encoder wrappers ----------
class TextEncoder:
    def encode(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


class BGETextEncoder(TextEncoder):
    def __init__(self, model_name: str, device: torch.device, batch_size: int):
        self.model = BGEM3FlagModel(model_name, use_fp16=device.type == "cuda")
        self.batch_size = batch_size

    def encode(self, texts: List[str]) -> np.ndarray:
        dense_vectors: List[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            outputs = self.model.encode(
                batch,
                batch_size=len(batch),
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            dense_vectors.extend(outputs["dense_vecs"])
        arr = np.asarray(dense_vectors, dtype=np.float32)
        return l2_normalize(arr)


class SentenceTransformerEncoder(TextEncoder):
    def __init__(self, model_name: str, device: torch.device, batch_size: int):
        self.model = SentenceTransformer(model_name, device=str(device))
        self.batch_size = batch_size

    def encode(self, texts: List[str]) -> np.ndarray:
        arr = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return arr.astype(np.float32)


def prepare_text_encoder(name: str, device: torch.device, batch_size: int) -> TextEncoder:
    """Factory for text encoders."""
    registry = {
        "bge-m3": lambda: BGETextEncoder("BAAI/bge-m3", device, batch_size),
        "minilm": lambda: SentenceTransformerEncoder("sentence-transformers/all-MiniLM-L6-v2", device, batch_size),
        "mpnet": lambda: SentenceTransformerEncoder("sentence-transformers/multi-qa-mpnet-base-dot-v1", device, batch_size),
    }
    if name not in registry:
        raise ValueError(f"Unknown text encoder '{name}'. Options: {list(registry.keys())}")
    return registry[name]()


# ---------- Data loading ----------
def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load post metadata and user->post likes from CSVs."""
    posts = pd.read_csv(POSTS_PATH).fillna("")
    likes = pd.read_csv(LIKES_PATH)
    return posts, likes


def prepare_image_models(device: torch.device) -> Tuple[AutoImageProcessor, AutoModel]:
    """Load image processor and model."""
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    model.eval()
    return processor, model


# ---------- Embedding ----------
def embed_text(posts: pd.DataFrame, text_encoder: TextEncoder) -> Dict[str, np.ndarray]:
    """Embed post text; skip rows without text."""
    embeddings: Dict[str, np.ndarray] = {}
    text_rows = posts[posts["text"].str.len() > 0]
    if text_rows.empty:
        return embeddings
    encoded = text_encoder.encode(text_rows["text"].tolist())
    for post_id, vector in zip(text_rows["post_id"].tolist(), encoded):
        embeddings[post_id] = vector
    return embeddings


def embed_images(
    posts: pd.DataFrame,
    processor: AutoImageProcessor,
    model: AutoModel,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Embed post images with Dinov2; skip rows without images."""
    embeddings: Dict[str, np.ndarray] = {}
    image_rows = posts[posts["image_path"].str.len() > 0]
    for _, row in image_rows.iterrows():
        image_path = DATA_ROOT / row["image_path"]
        if not image_path.exists():
            base = image_path.with_suffix("")
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = base.with_suffix(ext)
                if candidate.exists():
                    image_path = candidate
                    break
            else:
                print(f"Warning: missing image for post {row['post_id']} at {row['image_path']}; skipping.")
                continue
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            cls_token = outputs.last_hidden_state[:, 0, :]
        vector = cls_token.cpu().numpy()[0]
        embeddings[row["post_id"]] = vector
    return embeddings


def learn_image_projection(
    paired_image_embeddings: List[np.ndarray],
    paired_text_embeddings: List[np.ndarray],
) -> np.ndarray:
    """Solve a linear mapping so image vectors land in the text embedding space."""
    X = np.vstack(paired_image_embeddings)
    Y = np.vstack(paired_text_embeddings)
    lambda_reg = 1e-2
    XtX = X.T @ X
    XtY = X.T @ Y
    W = np.linalg.solve(XtX + lambda_reg * np.eye(XtX.shape[0]), XtY)
    return W  # shape: (image_dim, text_dim)


def fuse_post_embeddings(
    posts: pd.DataFrame,
    text_embeddings: Dict[str, np.ndarray],
    image_embeddings: Dict[str, np.ndarray],
    projection: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Combine text and projected image vectors into one per-post embedding."""
    fused: Dict[str, np.ndarray] = {}
    for _, row in posts.iterrows():
        post_id = row["post_id"]
        text_vec = text_embeddings.get(post_id)
        image_vec = image_embeddings.get(post_id)

        if text_vec is None and image_vec is None:
            continue

        if text_vec is not None and image_vec is not None:
            text_unit = l2_normalize(text_vec.reshape(1, -1))[0]
            image_unit = l2_normalize((image_vec @ projection).reshape(1, -1))[0]
            combined = 0.7 * text_unit + 0.3 * image_unit
        elif text_vec is not None:
            combined = text_vec
        else:
            combined = l2_normalize((image_vec @ projection).reshape(1, -1))[0]

        fused[post_id] = l2_normalize(combined.reshape(1, -1))[0]
    return fused


def build_user_embeddings(
    likes: pd.DataFrame,
    post_embeddings: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Average the embeddings of posts each user likes."""
    user_to_posts = likes.groupby("user_id")["post_id"].apply(list)
    user_embeddings: Dict[str, np.ndarray] = {}
    for user_id, post_ids in user_to_posts.items():
        vectors = [post_embeddings[pid] for pid in post_ids if pid in post_embeddings]
        if not vectors:
            continue
        stacked = np.vstack(vectors)
        user_embeddings[user_id] = l2_normalize(stacked.mean(axis=0, keepdims=True))[0]
    return user_embeddings


def build_faiss_index(post_embeddings: Dict[str, np.ndarray]) -> Tuple[faiss.IndexFlatIP, List[str]]:
    """Build a cosine-similarity FAISS index of post vectors."""
    if not post_embeddings:
        raise ValueError("No post embeddings to index.")
    dim = next(iter(post_embeddings.values())).shape[0]
    index = faiss.IndexFlatIP(dim)
    post_ids = list(post_embeddings.keys())
    matrix = np.vstack([post_embeddings[pid] for pid in post_ids]).astype("float32")
    index.add(matrix)
    return index, post_ids


def recommend_for_user(
    user_id: str,
    user_embeddings: Dict[str, np.ndarray],
    post_embeddings: Dict[str, np.ndarray],
    likes: pd.DataFrame,
    index: faiss.IndexFlatIP,
    post_ids: List[str],
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    """Return top_k unseen posts for the user with similarity scores."""
    if user_id not in user_embeddings:
        raise KeyError(f"User {user_id} not found.")

    liked_posts = set(likes[likes["user_id"] == user_id]["post_id"].tolist())
    query_vec = user_embeddings[user_id].astype("float32").reshape(1, -1)
    scores, neighbors = index.search(query_vec, top_k + len(liked_posts))

    recs: List[Tuple[str, float]] = []
    for idx, score in zip(neighbors[0], scores[0]):
        post_id = post_ids[int(idx)]
        if post_id in liked_posts:
            continue
        recs.append((post_id, float(score)))
        if len(recs) == top_k:
            break
    return recs


def format_post(row: pd.Series) -> str:
    """Readable one-liner for a post."""
    parts = [f"{row['post_id']} ({row['theme']})"]
    if isinstance(row["text"], str) and row["text"]:
        parts.append(f"text='{row['text']}'")
    if isinstance(row["image_path"], str) and row["image_path"]:
        parts.append(f"image={row['image_path']}")
    return " | ".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post recommender with pluggable text encoders.")
    parser.add_argument(
        "--text-encoders",
        default="bge-m3",
        help="Comma-separated list of text encoders to compare: bge-m3,minilm,mpnet",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for text encoding to avoid OOM.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many recommendations to show.",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default=None,
        help="Optional fixed user_id to evaluate; defaults to random per encoder.",
    )
    return parser.parse_args()


def demo() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoders = [name.strip() for name in args.text_encoders.split(",") if name.strip()]
    print(f"Using device: {device} | text encoders: {encoders}")

    posts, likes = load_datasets()
    image_processor, image_model = prepare_image_models(device)
    image_embeddings = embed_images(posts, image_processor, image_model, device)

    for enc_name in encoders:
        print("\n" + "=" * 60)
        print(f"Text encoder: {enc_name}")
        text_encoder = prepare_text_encoder(enc_name, device, args.batch_size)

        text_embeddings = embed_text(posts, text_encoder)

        paired_ids = [pid for pid in posts["post_id"] if pid in text_embeddings and pid in image_embeddings]
        paired_image_vectors = [image_embeddings[pid] for pid in paired_ids]
        paired_text_vectors = [text_embeddings[pid] for pid in paired_ids]
        projection = learn_image_projection(paired_image_vectors, paired_text_vectors)

        post_embeddings = fuse_post_embeddings(
            posts,
            text_embeddings,
            image_embeddings,
            projection,
        )
        user_embeddings = build_user_embeddings(likes, post_embeddings)
        index, post_ids = build_faiss_index(post_embeddings)

        if args.user_id:
            user_id = args.user_id
            if user_id not in user_embeddings:
                print(f"[warn] User {user_id} has no embedding for encoder {enc_name}; skipping.")
                continue
        else:
            random.seed(26)
            user_id = random.choice(list(user_embeddings.keys()))

        recs = recommend_for_user(user_id, user_embeddings, post_embeddings, likes, index, post_ids, top_k=args.top_k)

        liked_rows = posts[posts["post_id"].isin(likes[likes["user_id"] == user_id]["post_id"])]
        print(f"\nUser: {user_id}")
        print("Previously liked posts:")
        for _, row in liked_rows.iterrows():
            print(f"  - {format_post(row)}")

        print("\nRecommended posts (cosine similarity):")
        for post_id, score in recs:
            row = posts[posts["post_id"] == post_id].iloc[0]
            print(f"  - {format_post(row)} | score={score:.3f}")


if __name__ == "__main__":
    demo()
