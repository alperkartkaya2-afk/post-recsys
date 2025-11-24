# Post Recommender Walkthrough

This project builds a tiny but end-to-end recommender that mixes text and image signals in the same embedding space so we can serve posts similar to what a user already likes. Everything lives in simple CSVs and one Python script to keep things readable. The emphasis is on transparent mechanics: how embeddings are computed, aligned, fused, and searched.

## Data we feed the model
The dataset lives in CSVs under `data/` and is intentionally small so you can inspect it and reason about behavior.

```csv
# data/posts.csv (excerpt)
post_id,theme,text,image_path
p001,nature,Warm sunset over coastal mountains with soft orange light.,images/sunset.jpg
p002,nature,Pine forest trail after rain with mist between tall trees.,images/forest.jpg
...
p037,tech,Benchmark results comparing mobile GPUs for on-device AI.,images/p037.jpg
```

```csv
# data/user_likes.csv (excerpt)
user_id,post_id
user_anna,p001
user_anna,p002
user_mo,p003
...
user_auto005,p091
```

Some posts are text-only, some image-only, many have both; users like posts from coherent themes to make recommendations meaningful.

## Embedding the content
We load the models and embed text and images. Image loading tolerates extension changes. The goal is to put both modalities into comparable vectors before alignment.

```python
def prepare_models(device: torch.device):
    text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=str(device))
    image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    image_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    image_model.eval()
    return text_model, image_processor, image_model

def embed_text(posts, text_model):
    text_rows = posts[posts["text"].str.len() > 0]
    encoded = text_model.encode(
        text_rows["text"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return {pid: vec for pid, vec in zip(text_rows["post_id"], encoded)}

def embed_images(posts, processor, model, device):
    embeddings = {}
    image_rows = posts[posts["image_path"].str.len() > 0]
    for _, row in image_rows.iterrows():
        image_path = DATA_DIR / row["image_path"]
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
        embeddings[row["post_id"]] = cls_token.cpu().numpy()[0]
    return embeddings
```
Explanations:
- `normalize_embeddings=True` gives unit-length text vectors, simplifying cosine computations and keeping magnitudes consistent across batches.
- The image encoder uses the CLS token as a global descriptor; this is standard for ViT-like architectures and captures holistic content.
- Extension fallback avoids brittle paths; warnings keep you aware of missing assets without crashing the pipeline.

## Aligning text and images
We learn a linear map so image vectors land in the text space, then fuse modalities per post. Text gets slightly higher weight; image-only posts use the projected image alone. This aligns heterogeneous data without a heavy training loop.

```python
def learn_image_projection(paired_image_embeddings, paired_text_embeddings):
    X = np.vstack(paired_image_embeddings)
    Y = np.vstack(paired_text_embeddings)
    lambda_reg = 1e-2
    XtX = X.T @ X
    XtY = X.T @ Y
    return np.linalg.solve(XtX + lambda_reg * np.eye(XtX.shape[0]), XtY)

def fuse_post_embeddings(posts, text_embeddings, image_embeddings, projection):
    fused = {}
    for _, row in posts.iterrows():
        pid = row["post_id"]
        text_vec = text_embeddings.get(pid)
        image_vec = image_embeddings.get(pid)
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
        fused[pid] = l2_normalize(combined.reshape(1, -1))[0]
    return fused
```
Explanations:
- The projection solves a ridge regression: `W = argmin ||XW - Y||^2 + λ||W||^2`. λ=1e-2 stabilizes the inverse `XtX` when pairs are few.
- Fusion uses a convex combination; weighting text higher anchors semantics while the projected image adjusts similarity for visual cues.
- A second L2-normalization after fusion ensures every post vector lies on the unit sphere, making FAISS inner-product behave as cosine.

## Building user and retrieval embeddings
Users are the mean of what they liked; FAISS does fast cosine search and we filter out already-liked posts. This is a centroid-based collaborative filtering approach that works well for sparse implicit likes.

```python
def build_user_embeddings(likes, post_embeddings):
    user_embeddings = {}
    for user_id, post_ids in likes.groupby("user_id")["post_id"].apply(list).items():
        vectors = [post_embeddings[pid] for pid in post_ids if pid in post_embeddings]
        if vectors:
            user_embeddings[user_id] = l2_normalize(np.vstack(vectors).mean(axis=0, keepdims=True))[0]
    return user_embeddings

def build_faiss_index(post_embeddings):
    dim = next(iter(post_embeddings.values())).shape[0]
    index = faiss.IndexFlatIP(dim)
    post_ids = list(post_embeddings.keys())
    matrix = np.vstack([post_embeddings[pid] for pid in post_ids]).astype("float32")
    index.add(matrix)
    return index, post_ids

def recommend_for_user(user_id, user_embeddings, post_embeddings, likes, index, post_ids, top_k=3):
    liked_posts = set(likes[likes["user_id"] == user_id]["post_id"].tolist())
    query_vec = user_embeddings[user_id].astype("float32").reshape(1, -1)
    scores, neighbors = index.search(query_vec, top_k + len(liked_posts))
    recs = []
    for idx, score in zip(neighbors[0], scores[0]):
        pid = post_ids[int(idx)]
        if pid in liked_posts:
            continue
        recs.append((pid, float(score)))
        if len(recs) == top_k:
            break
    return recs
```
Explanations:
- User embedding = centroid of liked posts; this treats preferences as a point on the unit sphere and smooths over individual item noise.
- FAISS `IndexFlatIP` is exact (no quantization). With unit vectors, inner product equals cosine, so retrieval is consistent with our similarity measure.
- The recommender over-fetches (`top_k + len(liked)`) to ensure filtering does not shrink the result set; this is important when users have many likes.

## Running the demo
Steps:
1) Install deps: `pip3 install -r requirements.txt`
2) Generate or refresh data if needed: `python3 data/generate_mock_data.py` (optional if you want the base set)
3) Run the recommender: `python3 src/recommender.py`

Core demo loop:

```python
def demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    posts, likes = load_datasets()
    text_model, image_processor, image_model = prepare_models(device)
    text_embeddings = embed_text(posts, text_model)
    image_embeddings = embed_images(posts, image_processor, image_model, device)
    paired_ids = [pid for pid in posts["post_id"] if pid in text_embeddings and pid in image_embeddings]
    projection = learn_image_projection(
        [image_embeddings[pid] for pid in paired_ids],
        [text_embeddings[pid] for pid in paired_ids],
    )
    post_embeddings = fuse_post_embeddings(posts, text_embeddings, image_embeddings, projection)
    user_embeddings = build_user_embeddings(likes, post_embeddings)
    index, post_ids = build_faiss_index(post_embeddings)
    user_id = random.choice(list(user_embeddings.keys()))
    recs = recommend_for_user(user_id, user_embeddings, post_embeddings, likes, index, post_ids, top_k=3)
    ...
```
Explanations:
- Random user selection surfaces diverse cases; swap to a fixed `user_id` for reproducibility in experiments.
- The path from raw CSV → embeddings → projection → fusion → user profiles → FAISS index mirrors a real offline pipeline; each step can be swapped (e.g., different encoders, different fusion weights) without touching the rest.
- Everything runs in-memory to keep the loop fast; for larger data you’d batch encoding and persist embeddings to disk before indexing.

Sample output from the current dataset:

```
User: user_lee
Previously liked posts:
  - p019 (nature) | text='Hidden waterfall in a mossy canyon with cool mist.' | image=images/p019.jpg
  - p020 (nature) | text='Wildflower meadow below snow-capped peaks during sunrise.' | image=images/p020.jpg

Recommended posts (cosine similarity):
  - p028 (nature) | image=images/p028.jpg | score=0.692
  - p027 (nature) | image=images/p027.jpg | score=0.672
  - p014 (nature) | text='Snowy ridge hike with bright alpine air and distant peaks.' | image=images/p014.jpg | score=0.535
```

## Why this design is simple and scalable
- Single-pass embedding and alignment: we avoid iterative training; a closed-form projection keeps complexity low and reproducibility high.
- Modularity: logic lives in `src/recommender.py`; data is plain CSV you can edit or regenerate, keeping experiments cheap.
- Extensible knobs: adjust fusion weights, swap encoders, or add posts/users without changing retrieval logic.
- Fast retrieval: FAISS `IndexFlatIP` is simple, scales linearly, and can move to GPU later; cosine is just inner product on unit vectors.

## Next small steps
- Add more paired text-image posts to strengthen the image→text projection and reduce off-theme matches.
- Swap Dinov2 for a CLIP-style model to compare cross-modal alignment quality.
- Log recall@k on a holdout set of likes for a quantifiable metric instead of the printed sample.
