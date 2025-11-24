"""
Simple post recommender that mixes text and image signals into one embedding space.

Workflow:
- load tiny CSV datasets from data/posts.csv and data/user_likes.csv
- embed text with sentence-transformers and images with Dinov2
- learn a linear projection to map image features into the text embedding space
- fuse per-post embeddings, average liked posts to build user profiles
- build a FAISS index for fast similarity search and demo a recommendation
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoImageProcessor, AutoModel

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def l2_normalize(array: np.ndarray) -> np.ndarray:
    """Normalize rows to unit length."""
    norms = np.linalg.norm(array, axis=1, keepdims=True) + 1e-12
    return array / norms


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load post metadata and user->post likes from CSVs."""
    posts = pd.read_csv(DATA_DIR / "posts.csv").fillna("")
    likes = pd.read_csv(DATA_DIR / "user_likes.csv")
    return posts, likes


def prepare_models(device: torch.device) -> Tuple[SentenceTransformer, AutoImageProcessor, AutoModel]:
    """Load text and image encoders."""
    text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=str(device))
    image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    image_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    image_model.eval()
    return text_model, image_processor, image_model


def embed_text(posts: pd.DataFrame, text_model: SentenceTransformer) -> Dict[str, np.ndarray]:
    """Embed post text; skip rows without text."""
    embeddings: Dict[str, np.ndarray] = {}
    text_rows = posts[posts["text"].str.len() > 0]
    if text_rows.empty:
        return embeddings
    encoded = text_model.encode(
        text_rows["text"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
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
        image_path = DATA_DIR / row["image_path"]
        if not image_path.exists():
            base = image_path.with_suffix("")
            # Try common alternate extensions so we tolerate .png/.jpg swaps.
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
            cls_token = outputs.last_hidden_state[:, 0, :]  # take CLS token representation
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
    # Minimize ||X W - Y||_2 with a tiny ridge term for stability.
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
            # Favor text slightly to keep recommendations grounded in language.
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
    # Over-fetch to account for filtering liked items.
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


def demo() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    posts, likes = load_datasets()
    text_model, image_processor, image_model = prepare_models(device)

    text_embeddings = embed_text(posts, text_model)
    image_embeddings = embed_images(posts, image_processor, image_model, device)

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

    random.seed(26)
    user_id = random.choice(list(user_embeddings.keys()))
    recs = recommend_for_user(user_id, user_embeddings, post_embeddings, likes, index, post_ids, top_k=3)

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
