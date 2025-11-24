"""
Automate small dataset growth while keeping user tastes coherent.

What it does:
- Reads existing posts.csv and user_likes.csv
- Infers each user's dominant theme from their likes
- Generates new posts per user that match their theme so evaluations stay meaningful
- (Optional) Downloads a keyword-based image per new post into data/images (Unsplash/Picsum)
- Appends the new posts and likes back to the CSVs

Usage examples:
- Dry run only (see what would be added): python3 data/auto_augment_dataset.py --posts-per-user 1
- Actually write rows, skip image downloads: python3 data/auto_augment_dataset.py --posts-per-user 1 --write
- Write rows and download images with throttling: python3 data/auto_augment_dataset.py --posts-per-user 10 --write --download-images --download-delay 1.0
"""
from __future__ import annotations

import argparse
import itertools
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import quote_plus

import pandas as pd
import requests


DATA_DIR = Path(__file__).parent
POSTS_PATH = DATA_DIR / "posts.csv"
LIKES_PATH = DATA_DIR / "user_likes.csv"
IMAGES_DIR = DATA_DIR / "images"
MAX_IMAGES_PER_USER = 2  # when creating new posts, how many should have images

# Small pools of on-theme text snippets to keep new samples coherent.
THEME_SNIPPETS: Dict[str, List[str]] = {
    "nature": [
        "Golden alpine meadow with late-summer wildflowers and distant peaks.",
        "Rocky coastline with fog rolling over turquoise water.",
        "Glacial lake at dawn reflecting snow-capped mountains.",
        "Dense pine forest after rain with mist along the trail.",
        "Canyon slot lit by a narrow beam of sunlight and dust.",
        "High-altitude ridge hike above a sea of clouds.",
        "Waterfall plunging into a mossy gorge surrounded by ferns.",
        "Aurora streaks over a frozen lake and silhouetted trees.",
    ],
    "art": [
        "Bold neon graffiti spanning a brick underpass with geometric shapes.",
        "Ink wash skyline sketch with a faint sunset gradient.",
        "Collage of torn paper textures in pastel tones.",
        "Minimalist continuous-line portrait on off-white paper.",
        "Abstract acrylic pour with swirling magenta and cyan.",
        "Spray paint mural wrapping subway tiles with arrows and tags.",
        "Charcoal study of faces with dramatic shading.",
        "Mid-century inspired color-block poster with clean typography.",
    ],
    "tech": [
        "Hands-on build log for a tiny single-board computer cluster.",
        "Wireless mechanical keyboard mod with custom keycaps.",
        "Benchmarking on-device AI performance across mobile GPUs.",
        "Deploying a quantized model with ONNX Runtime on Raspberry Pi.",
        "Reviewing ultrabook battery life while coding on the go.",
        "Setting up a tidy dual-monitor desk with cable management.",
        "Notebook on vector search performance CPU vs GPU.",
        "Exploring low-latency edge inference with mixed precision.",
    ],
    "food": [
        "Sourdough toast topped with roasted veggies and herbs.",
        "Blueberry pancakes stacked with maple drizzle and butter.",
        "Iced matcha latte beside a flaky almond croissant.",
        "Spicy ramen with soft egg, chili oil, and scallions.",
        "Chocolate lava cake plated with berries and vanilla ice cream.",
        "Fresh salsa with avocado, lime, and crunchy tortilla chips.",
        "Crispy rosemary potatoes with sea salt flakes.",
        "Heirloom tomato caprese with basil oil and burrata.",
    ],
}


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    posts = pd.read_csv(POSTS_PATH).fillna("")
    likes = pd.read_csv(LIKES_PATH)
    return posts, likes


def infer_user_themes(posts: pd.DataFrame, likes: pd.DataFrame) -> Dict[str, str]:
    """Pick each user's dominant theme based on their liked posts."""
    theme_by_post = posts.set_index("post_id")["theme"].to_dict()
    user_theme: Dict[str, str] = {}
    for user_id, group in likes.groupby("user_id"):
        counts: Dict[str, int] = {}
        for pid in group["post_id"]:
            theme = theme_by_post.get(pid)
            if not theme:
                continue
            counts[theme] = counts.get(theme, 0) + 1
        if counts:
            user_theme[user_id] = max(counts.items(), key=lambda kv: kv[1])[0]
    return user_theme


def next_user_ids(likes: pd.DataFrame, how_many: int) -> List[str]:
    """Generate new unique user IDs."""
    existing = set(likes["user_id"].tolist())
    users: List[str] = []
    idx = 1
    while len(users) < how_many:
        candidate = f"user_auto{idx:03d}"
        if candidate not in existing:
            users.append(candidate)
        idx += 1
    return users


def next_post_id(posts: pd.DataFrame):
    """Yield sequential post IDs beyond the current max (e.g., p014, p015)."""
    existing_nums = [
        int(pid[1:])
        for pid in posts["post_id"]
        if isinstance(pid, str) and pid.startswith("p") and pid[1:].isdigit()
    ]
    start = max(existing_nums, default=0) + 1
    for n in itertools.count(start):
        yield f"p{n:03d}"


def build_new_posts(
    user_theme: Dict[str, str],
    posts: pd.DataFrame,
    posts_per_user: int,
    require_images: int = MAX_IMAGES_PER_USER,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Create new post rows and matching user_like rows."""
    existing_texts = set(posts["text"].tolist())
    pid_gen = next_post_id(posts)
    new_posts: List[Dict[str, str]] = []
    new_likes: List[Dict[str, str]] = []

    for user_idx, (user_id, theme) in enumerate(user_theme.items()):
        snippets = THEME_SNIPPETS.get(theme, [])
        if not snippets:
            continue
        # Offset per-user so two users of the same theme get different snippets first.
        start = user_idx % len(snippets)
        image_budget = require_images
        for i in range(posts_per_user):
            chosen_text = None
            for step in range(len(snippets)):
                candidate = snippets[(start + i + step) % len(snippets)]
                if candidate not in existing_texts:
                    chosen_text = candidate
                    break
            if chosen_text is None:
                # Reuse a snippet with a suffix to keep uniqueness without blocking generation.
                candidate = snippets[(start + i) % len(snippets)]
                chosen_text = f"{candidate} ({user_id} #{i+1})"
            post_id = next(pid_gen)
            include_image = image_budget > 0
            image_path = f"images/{post_id}.jpg" if include_image else ""
            new_posts.append(
                {
                    "post_id": post_id,
                    "theme": theme,
                    "text": chosen_text,
                    "image_path": image_path,
                }
            )
            new_likes.append({"user_id": user_id, "post_id": post_id})
            existing_texts.add(chosen_text)
            if include_image:
                image_budget -= 1
    return new_posts, new_likes


def download_image_for_post(
    text: str,
    post_id: str,
    retries: int = 3,
    backoff: float = 2.0,
    use_fallbacks: bool = True,
    delay_seconds: float = 1.0,
    overwrite: bool = False,
) -> bool:
    """Fetch an image using lightweight keyword-based sources (no DuckDuckGo/Unsplash to avoid rate limits)."""
    query = text.split(".")[0].strip() or text or post_id

    out_path = IMAGES_DIR / f"{post_id}.jpg"
    if out_path.exists() and not overwrite:
        print(f"[skip] Image already exists for {post_id} at {out_path}")
        return True

    if not text:
        print(f"[warn] No text/theme available to search for {post_id}; skipping image.")
        return False

    candidate_urls: List[str] = [
        f"https://picsum.photos/seed/{quote_plus(query)}/600/400",
    ]
    if use_fallbacks:
        candidate_urls.append(f"https://loremflickr.com/600/400/{quote_plus(query)}")

    for idx, url in enumerate(candidate_urls, start=1):
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(resp.content)
            print(f"[ok] Saved image for {post_id} -> {out_path} (source {idx})")
            time.sleep(delay_seconds)
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Failed to download image for {post_id} from {url}: {exc}")
            if not use_fallbacks:
                break
            time.sleep(backoff)

    print(f"[warn] Giving up on image for {post_id} after trying {len(candidate_urls)} sources.")
    return False


def save_updates(posts: pd.DataFrame, likes: pd.DataFrame) -> None:
    posts.to_csv(POSTS_PATH, index=False)
    likes.to_csv(LIKES_PATH, index=False)
    print(f"[ok] Wrote {POSTS_PATH} and {LIKES_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand the mock recsys dataset with on-theme posts.")
    parser.add_argument("--posts-per-user", type=int, default=1, help="How many new posts to add per user.")
    parser.add_argument("--write", action="store_true", help="Persist changes to CSVs (default is dry run).")
    parser.add_argument(
        "--download-images",
        action="store_true",
        help="Download the first image result per new post into data/images.",
    )
    parser.add_argument(
        "--new-users",
        type=int,
        default=0,
        help="Create this many new users; each gets posts_per_user posts on one theme with 2 images.",
    )
    parser.add_argument(
        "--refresh-posts",
        nargs="+",
        help="Optional: list of existing post_ids to re-download images for (no new rows added).",
    )
    parser.add_argument(
        "--max-total-posts",
        type=int,
        default=0,
        help="Abort if total posts would exceed this number (0 disables the cap).",
    )
    parser.add_argument(
        "--download-delay",
        type=float,
        default=1.0,
        help="Seconds to sleep between image downloads to avoid rate limiting.",
    )
    parser.add_argument(
        "--overwrite-images",
        action="store_true",
        help="Redownload and overwrite existing images when downloading.",
    )
    args = parser.parse_args()

    random.seed(123)
    posts, likes = load_data()

    if args.refresh_posts:
        target_ids = set(args.refresh_posts)
        for _, row in posts.iterrows():
            if row["post_id"] not in target_ids:
                continue
            download_image_for_post(
                row["text"],
                row["post_id"],
                use_fallbacks=True,
                delay_seconds=args.download_delay,
                overwrite=args.overwrite_images,
            )
        return

    # Build which users to expand.
    if args.new_users > 0:
        user_theme: Dict[str, str] = {}
        theme_list = list(THEME_SNIPPETS.keys())
        new_user_ids = next_user_ids(likes, args.new_users)
        for idx, user_id in enumerate(new_user_ids):
            theme = theme_list[idx % len(theme_list)]
            user_theme[user_id] = theme
        print(f"Creating {len(new_user_ids)} new users with 3 posts each and {MAX_IMAGES_PER_USER} images per user.")
    else:
        user_theme = infer_user_themes(posts, likes)

    new_posts, new_likes = build_new_posts(
        user_theme,
        posts,
        args.posts_per_user,
        require_images=MAX_IMAGES_PER_USER,
    )

    if args.max_total_posts and len(posts) + len(new_posts) > args.max_total_posts:
        print(
            f"Cap reached: existing {len(posts)} + new {len(new_posts)} exceeds limit {args.max_total_posts}. Aborting."
        )
        return

    if not new_posts:
        print("No new posts generated (maybe snippets exhausted?).")
        return

    print(f"Planned additions: {len(new_posts)} posts, {len(new_likes)} user_like rows.")
    for row, like in zip(new_posts, new_likes):
        print(f"  - {row['post_id']} for {like['user_id']} [{row['theme']}] {row['text']}")

    failed_downloads: List[str] = []
    if args.download_images:
        for row in new_posts:
            ok = download_image_for_post(
                row["text"],
                row["post_id"],
                use_fallbacks=True,
                delay_seconds=args.download_delay,
                overwrite=args.overwrite_images,
            )
            if not ok:
                failed_downloads.append(row["post_id"])

    if failed_downloads:
        print(f"[warn] Images failed for: {', '.join(failed_downloads)}")

    if not args.write:
        print("\nDry run complete. Re-run with --write to persist changes.")
        return

    updated_posts = pd.concat([posts, pd.DataFrame(new_posts)], ignore_index=True)
    updated_likes = pd.concat([likes, pd.DataFrame(new_likes)], ignore_index=True)
    save_updates(updated_posts, updated_likes)


if __name__ == "__main__":
    main()
