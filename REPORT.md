# Post Recommender Walkthrough

This project builds a tiny but end-to-end recommender that mixes text and image signals in the same embedding space so we can serve posts that look like what a user already likes. Everything lives in simple CSVs and a single Python script to keep things readable.

## Data we feed the model
The dataset lives in CSVs under `data/`:

```csv
# data/posts.csv
post_id,theme,text,image_path
p001,nature,Warm sunset over coastal mountains with soft orange light.,images/sunset.jpg
p002,nature,Pine forest trail after rain with mist between tall trees.,images/forest.jpg
p003,art,Pastel abstract shapes arranged in a minimalist composition.,images/abstract.jpg
...
p013,food,Stack of pancakes with berries and maple syrup for brunch.,images/pancakes.jpg
```

```csv
# data/user_likes.csv
user_id,post_id
user_anna,p001
user_anna,p002
user_anna,p009
user_mo,p003
user_mo,p006
...
```

Themes (nature, art, tech, food) are coherent so we can see whether recommendations stay on-theme. Some posts are text-only, some image-only, and some have both so we can align modalities.

## Embedding the content
We load the models and embed text, themes, and images. Image loading tolerates `.jpg/.jpeg/.png` swaps to stay robust to filename changes.

```python
def prepare_models(device: torch.device) -> Tuple[SentenceTransformer, AutoImageProcessor, AutoModel]:
    text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=str(device))
    image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    image_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    image_model.eval()
    return text_model, image_processor, image_model

def embed_text(posts: pd.DataFrame, text_model: SentenceTransformer) -> Dict[str, np.ndarray]:
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

def embed_themes(posts: pd.DataFrame, text_model: SentenceTransformer) -> Dict[str, np.ndarray]:
    themes = sorted(posts["theme"].dropna().unique().tolist())
    if not themes:
        return {}
    encoded = text_model.encode(
        [f"{theme} content" for theme in themes],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return dict(zip(themes, encoded))

def embed_images(
    posts: pd.DataFrame,
    processor: AutoImageProcessor,
    model: AutoModel,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    embeddings: Dict[str, np.ndarray] = {}
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
        vector = cls_token.cpu().numpy()[0]
        embeddings[row["post_id"]] = vector
    return embeddings
```

## Aligning text and images
We learn a linear map so image vectors land in the text space, then fuse modalities per post. Text gets slightly higher weight; image-only posts borrow a theme embedding so they still have language grounding.

```python
def learn_image_projection(
    paired_image_embeddings: List[np.ndarray],
    paired_text_embeddings: List[np.ndarray],
) -> np.ndarray:
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
    theme_embeddings: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    fused: Dict[str, np.ndarray] = {}
    for _, row in posts.iterrows():
        post_id = row["post_id"]
        text_vec = text_embeddings.get(post_id)
        if text_vec is None:
            text_vec = theme_embeddings.get(row["theme"])
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
            combined = 0.4 * l2_normalize((image_vec @ projection).reshape(1, -1))[0]
            if row["theme"] in theme_embeddings:
                combined += 0.6 * theme_embeddings[row["theme"]]

        fused[post_id] = l2_normalize(combined.reshape(1, -1))[0]
    return fused
```

## Building user and retrieval embeddings
Users are the mean of what they liked; FAISS does fast cosine search and we filter out already-liked posts.

```python
def build_user_embeddings(
    likes: pd.DataFrame,
    post_embeddings: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
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
```

## Running the demo
Steps:
1. Install deps: `pip3 install -r requirements.txt`
2. Run the recommender: `python3 src/recommender.py`

Core demo loop:

```python
def demo() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    posts, likes = load_datasets()
    text_model, image_processor, image_model = prepare_models(device)

    text_embeddings = embed_text(posts, text_model)
    theme_embeddings = embed_themes(posts, text_model)
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
        theme_embeddings,
    )
    user_embeddings = build_user_embeddings(likes, post_embeddings)
    index, post_ids = build_faiss_index(post_embeddings)

    random.seed(42)
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
```

Current sample output:

```
User: user_anna
Previously liked posts:
  - p001 (nature) | text='Warm sunset over coastal mountains with soft orange light.' | image=images/sunset.jpg
  - p002 (nature) | text='Pine forest trail after rain with mist between tall trees.' | image=images/forest.jpg
  - p009 (nature) | text='Quiet beach morning with soft waves and pale blue sky.' | image=images/beach.jpg

Recommended posts (cosine similarity):
  - p010 (nature) | text='High-altitude lake reflecting snowy peaks and alpine sky.' | image=images/alpine.jpg | score=0.443
  - p007 (food) | image=images/bread.jpg | score=0.307
  - p004 (tech) | text='Review of the latest smartphone camera and night mode performance.' | score=0.293
```

## Why this design is simple and scalable
- Single-pass embedding: no heavy training loop, just one linear solve for cross-modal alignment.
- Modularity: all logic lives in `src/recommender.py`; data is plain CSV you can edit.
- Extensible knobs: change weights or add posts without touching retrieval code.
- Fast retrieval: FAISS `IndexFlatIP` scales linearly and can move to GPU later.

## Next small steps
- Add more paired text-image posts to strengthen the projection and reduce off-theme matches.
- Try a CLIP-style model for a pretrained cross-modal space and compare results.
- Log per-user recall@k on a holdout set for a quantifiable metric instead of the printed sample.
