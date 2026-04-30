"""
data_loader.py — Loader for Kaggle's "The Movies Dataset" (rounakbanik).

ID join chain (critical — skipping this produces wrong recommendations):
    ratings.movieId  →  links.movieId  →  links.tmdbId  →  movies_metadata.id

Required files in data/:
    movies_metadata.csv
    ratings.csv  (or ratings_small.csv in --small / dev mode)
    links.csv    (or links_small.csv)
    keywords.csv  (optional — enriches content-based model)
    credits.csv   (optional — enriches content-based model)
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Wire tqdm into pandas .progress_apply() once
tqdm.pandas(desc="Parsing", unit="row")


# ── Helpers ─────────────────────────────────────────────────────────────────

def _parse_json_col(val: str, name_key: str = "name") -> list[str]:
    """'[{"id":18,"name":"Drama"},...]'  →  ['Drama', ...]"""
    try:
        items = ast.literal_eval(str(val))
        if isinstance(items, list):
            return [d[name_key] for d in items if isinstance(d, dict) and name_key in d]
    except (ValueError, SyntaxError):
        pass
    return []


def _parse_director(crew_val: str) -> str:
    """Extract director name from credits crew JSON string."""
    try:
        for member in ast.literal_eval(str(crew_val)):
            if isinstance(member, dict) and member.get("job") == "Director":
                return str(member.get("name", ""))
    except (ValueError, SyntaxError):
        pass
    return ""


# ── Main loader ──────────────────────────────────────────────────────────────

def load_data(
    data_dir: str = "data",
    small: bool = False,
    nrows: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and merge all relevant Kaggle dataset files.

    Parameters
    ----------
    data_dir : folder containing the CSV files
    small    : use ratings_small.csv + links_small.csv (fast dev mode)
    nrows    : cap the number of rating rows loaded (None = all)

    Returns
    -------
    ratings : DataFrame[userId, movieId, rating, timestamp]
              movieId is the TMDB id after the ML→TMDB remap.
    movies  : DataFrame[id, title, genres, genre_str, content, overview,
                        vote_average, vote_count, keyword_str, cast_str, director]
    """
    data_path    = Path(data_dir)
    ratings_file = data_path / ("ratings_small.csv" if small else "ratings.csv")
    links_file   = data_path / ("links_small.csv"   if small else "links.csv")

    # ── 1. Ratings ───────────────────────────────────────────────────────────
    logger.info("[1/6] Loading ratings from %s...", ratings_file)
    ratings = pd.read_csv(
        ratings_file, nrows=nrows,
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
    )
    logger.info(
        "      %d ratings | %d users", len(ratings), ratings["userId"].nunique()
    )

    # ── 2. Remap MovieLens → TMDB ids ────────────────────────────────────────
    logger.info("[2/6] Remapping MovieLens IDs → TMDB IDs via %s...", links_file)
    links = pd.read_csv(links_file, dtype={"movieId": "int32"})
    links = links.dropna(subset=["tmdbId"])
    links["tmdbId"] = links["tmdbId"].astype("int32")
    ml_to_tmdb: dict[int, int] = links.set_index("movieId")["tmdbId"].to_dict()

    before  = len(ratings)
    ratings = ratings[ratings["movieId"].isin(ml_to_tmdb)].copy()
    ratings["movieId"] = ratings["movieId"].map(ml_to_tmdb)
    logger.info("      Kept %d/%d ratings after join", len(ratings), before)

    # ── 3. movies_metadata ───────────────────────────────────────────────────
    logger.info("[3/6] Loading movies_metadata.csv...")
    meta_cols = ["id", "title", "genres", "overview", "vote_average", "vote_count", "release_date"]
    movies = pd.read_csv(data_path / "movies_metadata.csv", usecols=meta_cols, low_memory=False)
    movies = movies[pd.to_numeric(movies["id"], errors="coerce").notna()].copy()
    movies["id"] = movies["id"].astype("int32")
    movies = movies.drop_duplicates(subset="id").reset_index(drop=True)
    logger.info("      %d movies loaded", len(movies))

    tqdm.pandas(desc="  genres", unit="row")
    movies["genre_list"] = movies["genres"].progress_apply(_parse_json_col)
    movies["genre_str"]  = movies["genre_list"].apply(" ".join)
    movies["genres"]     = movies["genre_str"]

    # ── 4. Keywords ──────────────────────────────────────────────────────────
    logger.info("[4/6] Loading keywords.csv...")
    try:
        kw = pd.read_csv(data_path / "keywords.csv")
        kw["id"] = pd.to_numeric(kw["id"], errors="coerce")
        kw = kw.dropna(subset=["id"])
        kw["id"] = kw["id"].astype("int32")

        tqdm.pandas(desc="  keywords", unit="row")
        kw["keyword_str"] = kw["keywords"].progress_apply(_parse_json_col).apply(" ".join)
        movies = movies.merge(kw[["id", "keyword_str"]], on="id", how="left")
        logger.info(
            "      %d keywords parsed",
            kw["keyword_str"].str.split().apply(len).sum(),
        )
    except FileNotFoundError:
        logger.warning("      keywords.csv not found — skipping")

    movies["keyword_str"] = movies.get(
        "keyword_str", pd.Series("", index=movies.index)
    ).fillna("")

    # ── 5. Credits ───────────────────────────────────────────────────────────
    logger.info("[5/6] Loading credits.csv...")
    try:
        credits = pd.read_csv(data_path / "credits.csv")
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
        logger.info("      Cast + director parsed for %d movies", len(credits))
    except FileNotFoundError:
        logger.warning("      credits.csv not found — skipping")

    movies["cast_str"] = movies.get(
        "cast_str", pd.Series("", index=movies.index)
    ).fillna("")
    movies["director"] = movies.get(
        "director", pd.Series("", index=movies.index)
    ).fillna("")

    # ── 6. Build TF-IDF content string ───────────────────────────────────────
    logger.info("[6/6] Building content strings...")
    movies["content"] = (
        movies["genre_str"].str.repeat(3)
        + " " + movies["director"].str.repeat(2)
        + " " + movies["keyword_str"]
        + " " + movies["cast_str"]
    ).str.strip()

    movies["vote_average"] = pd.to_numeric(movies["vote_average"], errors="coerce").fillna(0)
    movies["vote_count"]   = pd.to_numeric(movies["vote_count"],   errors="coerce").fillna(0)
    movies["overview"]     = movies["overview"].fillna("")

    # Keep only movies that appear in ratings
    valid_ids = set(ratings["movieId"].unique())
    movies    = movies[movies["id"].isin(valid_ids)].reset_index(drop=True)

    logger.info("Ready: %d ratings | %d movies", len(ratings), len(movies))
    return ratings, movies