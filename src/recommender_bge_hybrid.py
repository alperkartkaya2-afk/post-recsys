"""
Post recommender variant that uses BGE-M3 for text embeddings (dense + sparse + ColBERT)
and Dinov2 for image embeddings, then blends all three text signals with projected image
features to rank recommendations.

Pipeline in plain English:
1) Load CSV data (posts, user likes).
2) Encode post text with BGE-M3, asking for all outputs:
   - dense pooled vector per text
   - sparse lexical weights (SPLADE-style)
   - token-level ColBERT embeddings
3) Encode post images with Dinov2 (CLS token).
4) Learn a linear projection so image vectors land in the dense text space.
5) Fuse text + projected image into one dense vector per post.
6) Build user profiles that aggregate:
   - dense fused vectors (mean of liked posts)
   - sparse tokens (sum of liked posts, then normalize)
   - ColBERT token matrices (list of liked posts' tokens)
7) Score every unseen post with a weighted blend of:
   dense cosine + sparse cosine + ColBERT late interaction.
8) Print top recommendations with each component score visible.
"""
from __future__ import annotations

import random
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from FlagEmbedding import BGEM3FlagModel
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BATCH_SIZE = 16  # text embedding batch size to avoid OOM


@dataclass
class TextEmbeddingResult:
    """
    Container for BGE outputs keyed by post_id.
    - dense: pooled sentence embedding
    - sparse: lexical weights as a normalized {token_id: weight} map
    - colbert: token-level embeddings (num_tokens x dim)
    """

    dense: Dict[str, np.ndarray] = field(default_factory=dict)
    sparse: Dict[str, Dict[int, float]] = field(default_factory=dict)  # normalized sparse dict
    colbert: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class UserProfile:
    """
    All three modalities merged for one user:
    - dense: mean of fused post vectors the user liked
    - sparse: summed lexical weights of liked posts, normalized
    - colbert: list of token matrices (one per liked post) for late interaction
    """

    dense: np.ndarray
    sparse: Dict[int, float]
    colbert: List[np.ndarray]


def l2_normalize(array: np.ndarray) -> np.ndarray:
    """Normalize rows to unit length."""
    norms = np.linalg.norm(array, axis=1, keepdims=True) + 1e-12
    return array / norms


def sparse_to_dict(sparse_vec: Any) -> Dict[int, float]:
    """
    Convert BGE sparse output into a simple {token_id: weight} dict.
    Handles both dict outputs (indices/values) and scipy sparse matrices.
    """
    if isinstance(sparse_vec, dict) and "indices" in sparse_vec and "values" in sparse_vec:
        indices = sparse_vec["indices"]
        values = sparse_vec["values"]
        if hasattr(indices, "tolist"):
            indices = indices.tolist()
        if hasattr(values, "tolist"):
            values = values.tolist()
        return {int(i): float(v) for i, v in zip(indices, values)}
    if hasattr(sparse_vec, "tocoo"):
        coo = sparse_vec.tocoo()
        return {int(j): float(v) for j, v in zip(coo.col, coo.data)}
    return {}


def normalize_sparse(vec: Dict[int, float]) -> Dict[int, float]:
    """
    L2-normalize a sparse dict so cosine similarity is simply dot product.
    """
    norm = math.sqrt(sum(v * v for v in vec.values())) + 1e-12
    return {k: v / norm for k, v in vec.items()}


def sparse_cosine(a: Dict[int, float], b: Dict[int, float]) -> float:
    """
    Cosine for pre-normalized sparse dicts (dot product over intersection).
    """
    if not a or not b:
        return 0.0
    # iterate over smaller map for speed
    if len(a) > len(b):
        a, b = b, a
    return float(sum(v * b.get(k, 0.0) for k, v in a.items()))


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load post metadata and user->post likes from CSVs."""
    posts = pd.read_csv(DATA_DIR / "posts.csv").fillna("")
    likes = pd.read_csv(DATA_DIR / "user_likes.csv")
    return posts, likes


def prepare_models(device: torch.device) -> Tuple[BGEM3FlagModel, AutoImageProcessor, AutoModel]:
    """
    Load text and image encoders.
    - BGE-M3 for text (dense + sparse + ColBERT)
    - Dinov2 for images (CLS token as image embedding)
    """
    text_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=device.type == "cuda")
    image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    image_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    image_model.eval()
    return text_model, image_processor, image_model


def embed_text(
    posts: pd.DataFrame,
    text_model: BGEM3FlagModel,
    batch_size: int = BATCH_SIZE,
) -> TextEmbeddingResult:
    """
    Embed post text with BGE-M3 in batches and return dense, sparse, and ColBERT outputs.
    """
    embeddings = TextEmbeddingResult()
    text_rows = posts[posts["text"].str.len() > 0]
    if text_rows.empty:
        return embeddings

    texts = text_rows["text"].tolist()
    ids = text_rows["post_id"].tolist()

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batch_ids = ids[start : start + batch_size]
        outputs = text_model.encode(
            batch_texts,
            batch_size=len(batch_texts),
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
        )

        # Dense: pooled vector per text (used for projection/fusion + dense retrieval).
        for post_id, vector in zip(batch_ids, outputs.get("dense_vecs", [])):
            vec = np.asarray(vector, dtype=np.float32)
            embeddings.dense[post_id] = l2_normalize(vec.reshape(1, -1))[0]

        # Sparse: lexical weights for hybrid scoring (if text exists).
        if "sparse_vecs" in outputs:
            for post_id, sparse_vec in zip(batch_ids, outputs["sparse_vecs"]):
                sparse_dict = normalize_sparse(sparse_to_dict(sparse_vec))
                if sparse_dict:
                    embeddings.sparse[post_id] = sparse_dict

        # ColBERT: token-level embeddings for late interaction reranking.
        if "colbert_vecs" in outputs:
            for post_id, colbert_vec in zip(batch_ids, outputs["colbert_vecs"]):
                if hasattr(colbert_vec, "cpu"):
                    colbert_vec = colbert_vec.cpu().numpy()
                embeddings.colbert[post_id] = np.asarray(colbert_vec, dtype=np.float32)

    return embeddings


def embed_images(
    posts: pd.DataFrame,
    processor: AutoImageProcessor,
    model: AutoModel,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Embed post images with Dinov2; skip rows without images.
    We keep only the CLS token as the image representation.
    """
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
    """
    Solve a linear mapping so image vectors land in the dense text embedding space.
    This lets us compare images to text by projecting images into the text space.
    """
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
    """
    Combine text and projected image vectors into one per-post embedding.
    - If both exist: weighted blend (text favored) then normalize.
    - If only one exists: normalize that one.
    """
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


def build_user_profiles(
    likes: pd.DataFrame,
    fused_embeddings: Dict[str, np.ndarray],
    text_sparse: Dict[str, Dict[int, float]],
    text_colbert: Dict[str, np.ndarray],
) -> Dict[str, UserProfile]:
    """
    Build per-user profiles that include:
      - dense: fused text+image vector
      - sparse: summed sparse tokens from liked posts (normalized)
      - colbert: token-level matrices from liked posts
    """
    user_to_posts = likes.groupby("user_id")["post_id"].apply(list)
    profiles: Dict[str, UserProfile] = {}
    for user_id, post_ids in user_to_posts.items():
        dense_vectors = [fused_embeddings[pid] for pid in post_ids if pid in fused_embeddings]
        if not dense_vectors:
            continue
        stacked = np.vstack(dense_vectors)
        dense_profile = l2_normalize(stacked.mean(axis=0, keepdims=True))[0]

        sparse_parts = [text_sparse[pid] for pid in post_ids if pid in text_sparse]
        sparse_profile: Dict[int, float] = {}
        for part in sparse_parts:
            for idx, val in part.items():
                sparse_profile[idx] = sparse_profile.get(idx, 0.0) + val
        if sparse_profile:
            sparse_profile = normalize_sparse(sparse_profile)

        colbert_parts = [text_colbert[pid] for pid in post_ids if pid in text_colbert]

        profiles[user_id] = UserProfile(
            dense=dense_profile,
            sparse=sparse_profile,
            colbert=colbert_parts,
        )
    return profiles


def colbert_score(query_vecs: np.ndarray, doc_vecs: np.ndarray) -> float:
    """
    ColBERT late-interaction score: max-sim per query token, then sum.
    Shapes: query_vecs [q_tokens, dim], doc_vecs [d_tokens, dim].
    """
    q = torch.tensor(query_vecs)
    d = torch.tensor(doc_vecs)
    sims = torch.matmul(q, d.T)
    return float(sims.max(dim=1).values.sum().item())


def recommend_for_user(
    user_id: str,
    user_profiles: Dict[str, UserProfile],
    fused_embeddings: Dict[str, np.ndarray],
    text_sparse: Dict[str, Dict[int, float]],
    text_colbert: Dict[str, np.ndarray],
    likes: pd.DataFrame,
    top_k: int = 3,
) -> List[Dict[str, float]]:
    """Return top_k unseen posts ranked by dense + sparse + ColBERT scores."""
    if user_id not in user_profiles:
        raise KeyError(f"User {user_id} not found.")

    profile = user_profiles[user_id]
    liked_posts = set(likes[likes["user_id"] == user_id]["post_id"].tolist())
    dense_weight = 0.5  # main semantic signal (fused text + projected image)
    sparse_weight = 0.2  # lexical grounding (only when text exists)
    colbert_weight = 0.3  # fine-grained token alignment (only when text exists)

    scored: List[Dict[str, float]] = []
    for post_id, dense_vec in fused_embeddings.items():
        if post_id in liked_posts:
            continue

        # Dense cosine (always available because fused embeddings exist for every item we consider).
        dense_score = float(np.dot(profile.dense, dense_vec))
        total = dense_score * dense_weight
        weight_sum = dense_weight

        # Sparse cosine (only if both user and post have sparse tokens).
        sparse_score = 0.0
        if profile.sparse and post_id in text_sparse:
            sparse_score = sparse_cosine(profile.sparse, text_sparse[post_id])
            total += sparse_score * sparse_weight
            weight_sum += sparse_weight

        # ColBERT late interaction (only if both sides have token matrices).
        colbert_score_val = 0.0
        if profile.colbert and post_id in text_colbert:
            doc_colbert = text_colbert[post_id]
            sims = [colbert_score(q, doc_colbert) for q in profile.colbert]
            if sims:
                # normalize by query length to keep scale comparable
                colbert_score_val = float(np.mean(sims) / max(doc_colbert.shape[0], 1))
                total += colbert_score_val * colbert_weight
                weight_sum += colbert_weight

        final_score = total / max(weight_sum, 1e-12)
        scored.append(
            {
                "post_id": post_id,
                "final_score": final_score,
                "dense_score": dense_score,
                "sparse_score": sparse_score,
                "colbert_score": colbert_score_val,
            }
        )

    scored.sort(key=lambda item: item["final_score"], reverse=True)
    return scored[:top_k]


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

    paired_ids = [pid for pid in posts["post_id"] if pid in text_embeddings.dense and pid in image_embeddings]
    paired_image_vectors = [image_embeddings[pid] for pid in paired_ids]
    paired_text_vectors = [text_embeddings.dense[pid] for pid in paired_ids]
    projection = learn_image_projection(paired_image_vectors, paired_text_vectors)

    post_embeddings = fuse_post_embeddings(
        posts,
        text_embeddings.dense,
        image_embeddings,
        projection,
    )
    user_profiles = build_user_profiles(likes, post_embeddings, text_embeddings.sparse, text_embeddings.colbert)

    random.seed(26)
    user_id = random.choice(list(user_profiles.keys()))
    recs = recommend_for_user(
        user_id,
        user_profiles,
        post_embeddings,
        text_embeddings.sparse,
        text_embeddings.colbert,
        likes,
        top_k=3,
    )

    liked_rows = posts[posts["post_id"].isin(likes[likes["user_id"] == user_id]["post_id"])]
    print(f"\nUser: {user_id}")
    print("Previously liked posts:")
    for _, row in liked_rows.iterrows():
        print(f"  - {format_post(row)}")

    print("\nRecommended posts (hybrid score | dense / sparse / colbert):")
    for rec in recs:
        row = posts[posts["post_id"] == rec["post_id"]].iloc[0]
        print(
            f"  - {format_post(row)} | final={rec['final_score']:.3f} "
            f"(dense={rec['dense_score']:.3f}, sparse={rec['sparse_score']:.3f}, colbert={rec['colbert_score']:.3f})"
        )


if __name__ == "__main__":
    demo()
