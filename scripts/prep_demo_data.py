"""
scripts/prep_demo_data.py — Generate slim artefacts for the Gradio demo.

Produces:
    data/movies_slim.parquet     (id, title, genre_str, content, overview)
                                  — built from the FULL movies catalog so every
                                    movie in the trained model's movie_map has a
                                    title (avoids "Unknown(tmdb=...)" in the UI)
    data/ratings_slim.parquet    (userId, movieId, rating)
                                  — ratings_small.csv only, keeps the demo fast

Why two separate loads?
    The trained model is built on the full 26M-rating dataset and its
    movie_map contains ~21k TMDB ids.  If movies_slim is generated from
    ratings_small.csv (~100k ratings, ~9k movies), most of those ids have
    no title entry and the UI shows "Unknown(tmdb=...)".

    Fix: load movies from the full catalog (fast — only movies_metadata.csv
    + keywords/credits are read; the 26M ratings file is NOT loaded) and
    load ratings from ratings_small.csv separately.

Run once locally before pushing to HuggingFace Spaces:
    python scripts/prep_demo_data.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# Allow running from repo root or scripts/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from tqdm import tqdm

tqdm.pandas(desc="Parsing", unit="row")


# ── Helpers (duplicated from data_loader to avoid loading ratings) ────────────

def _parse_json_col(val: str, name_key: str = "name") -> list[str]:
    try:
        items = ast.literal_eval(str(val))
        if isinstance(items, list):
            return [d[name_key] for d in items if isinstance(d, dict) and name_key in d]
    except (ValueError, SyntaxError):
        pass
    return []


def _parse_director(crew_val: str) -> str:
    try:
        for member in ast.literal_eval(str(crew_val)):
            if isinstance(member, dict) and member.get("job") == "Director":
                return str(member.get("name", ""))
    except (ValueError, SyntaxError):
        pass
    return ""


def _build_movies(data_dir: Path) -> pd.DataFrame:
    """
    Load and process the full movies catalog without touching ratings.
    Mirrors the logic in data_loader.load_data steps 3-6.
    """
    print("[1/4] Loading movies_metadata.csv...")
    meta_cols = ["id", "title", "genres", "overview", "vote_average", "vote_count"]
    movies = pd.read_csv(data_dir / "movies_metadata.csv", usecols=meta_cols, low_memory=False)
    movies = movies[pd.to_numeric(movies["id"], errors="coerce").notna()].copy()
    movies["id"] = movies["id"].astype("int32")
    movies = movies.drop_duplicates(subset="id").reset_index(drop=True)
    print(f"    {len(movies):,} movies loaded")

    tqdm.pandas(desc="  genres  ", unit="row")
    movies["genre_list"] = movies["genres"].progress_apply(_parse_json_col)
    movies["genre_str"]  = movies["genre_list"].apply(" ".join)

    print("[2/4] Loading keywords.csv...")
    try:
        kw = pd.read_csv(data_dir / "keywords.csv")
        kw["id"] = pd.to_numeric(kw["id"], errors="coerce")
        kw = kw.dropna(subset=["id"])
        kw["id"] = kw["id"].astype("int32")
        tqdm.pandas(desc="  keywords", unit="row")
        kw["keyword_str"] = kw["keywords"].progress_apply(_parse_json_col).apply(" ".join)
        movies = movies.merge(kw[["id", "keyword_str"]], on="id", how="left")
    except FileNotFoundError:
        print("    keywords.csv not found — skipping")
    movies["keyword_str"] = movies.get("keyword_str", pd.Series("", index=movies.index)).fillna("")

    print("[3/4] Loading credits.csv...")
    try:
        credits = pd.read_csv(data_dir / "credits.csv")
        credits["id"] = pd.to_numeric(credits["id"], errors="coerce")
        credits = credits.dropna(subset=["id"])
        credits["id"] = credits["id"].astype("int32")
        tqdm.pandas(desc="  cast    ", unit="row")
        credits["cast_str"] = credits["cast"].progress_apply(
            lambda v: " ".join(_parse_json_col(v)[:3])
        )
        tqdm.pandas(desc="  director", unit="row")
        credits["director"] = credits["crew"].progress_apply(_parse_director)
        movies = movies.merge(credits[["id", "cast_str", "director"]], on="id", how="left")
    except FileNotFoundError:
        print("    credits.csv not found — skipping")
    movies["cast_str"] = movies.get("cast_str", pd.Series("", index=movies.index)).fillna("")
    movies["director"] = movies.get("director", pd.Series("", index=movies.index)).fillna("")

    print("[4/4] Building content strings...")
    movies["content"] = (
        movies["genre_str"].str.repeat(3)
        + " " + movies["director"].str.repeat(2)
        + " " + movies["keyword_str"]
        + " " + movies["cast_str"]
    ).str.strip()
    movies["overview"] = movies["overview"].fillna("")

    return movies[["id", "title", "genre_str", "content", "overview"]].copy()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generate slim parquet files for the Gradio demo.")
    parser.add_argument(
        "--small", action="store_true",
        help=(
            "Use ratings_small.csv + links_small.csv for the ratings slice "
            "(matches `python main.py --small`). "
            "Without this flag the full ratings.csv is used, which matches a "
            "model trained with `python main.py`."
        ),
    )
    args = parser.parse_args()

    data_dir     = Path("data")
    ratings_file = "ratings_small.csv" if args.small else "ratings.csv"
    links_file   = "links_small.csv"   if args.small else "links.csv"
    mode_label   = "small" if args.small else "full"

    print(f"Mode: {mode_label}  (ratings={ratings_file}, links={links_file})")
    print()

    # ── Movies — always built from the full catalog ────────────────────────
    movies_slim = _build_movies(data_dir)

    # ── Ratings slice ─────────────────────────────────────────────────────
    print(f"Loading {ratings_file}...")
    ratings = pd.read_csv(
        data_dir / ratings_file,
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
    )
    print(f"    {len(ratings):,} ratings loaded")

    # Remap MovieLens IDs → TMDB IDs (same join as data_loader)
    print(f"Remapping via {links_file}...")
    links = pd.read_csv(data_dir / links_file, dtype={"movieId": "int32"})
    links = links.dropna(subset=["tmdbId"])
    links["tmdbId"] = links["tmdbId"].astype("int32")
    ml_to_tmdb = links.set_index("movieId")["tmdbId"].to_dict()
    before  = len(ratings)
    ratings = ratings[ratings["movieId"].isin(ml_to_tmdb)].copy()
    ratings["movieId"] = ratings["movieId"].map(ml_to_tmdb)
    print(f"    Kept {len(ratings):,}/{before:,} ratings after join")
    ratings_slim = ratings[["userId", "movieId", "rating"]].copy()

    # ── Build TF-IDF matrix offline so app.py startup is instant ─────────────
    print("[5/5] Building TF-IDF matrix (slow step — runs once offline)...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    import scipy.sparse
    import numpy as np

    text_col = "content" if "content" in movies_slim.columns else "genre_str"
    corpus   = movies_slim[text_col].fillna("").tolist()
    vec      = TfidfVectorizer(stop_words="english", max_features=20_000)
    matrix   = vec.fit_transform(corpus).astype(np.float32)

    out_tfidf = data_dir / "tfidf.npz"
    scipy.sparse.save_npz(str(out_tfidf), matrix)
    print(f"    TF-IDF shape: {matrix.shape}  nnz: {matrix.nnz:,}")

    # ── Save ──────────────────────────────────────────────────────────────
    out_movies  = data_dir / "movies_slim.parquet"
    out_ratings = data_dir / "ratings_slim.parquet"

    movies_slim.to_parquet(out_movies,  index=False)
    ratings_slim.to_parquet(out_ratings, index=False)

    print(f"\n✅ Saved {len(movies_slim):,} movies  → {out_movies}")
    print(f"✅ Saved {len(ratings_slim):,} ratings → {out_ratings}")
    print(f"✅ Saved TF-IDF matrix         → {out_tfidf}")
    print("\nNext steps:")
    if args.small:
        print("  1. Train the model:  python main.py --small")
    else:
        print("  1. Train the model:  python main.py")
    print("  2. Upload to HF Hub:")
    print("       huggingface-cli upload <repo> models/mf.pkl models/mf.pkl")
    print("       huggingface-cli upload <repo> data/tfidf.npz data/tfidf.npz")
    print("  3. Push to Spaces:   git push hf main")


if __name__ == "__main__":
    main()