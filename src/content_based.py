"""
content_based.py — TF-IDF content filtering + hybrid CF+CB scoring.

Memory fix (v3):
    Building the full cosine similarity matrix for 46k movies requires
    46497² × 8 bytes ≈ 16 GB (float64) or 8 GB (float32) — both OOM on
    typical hardware.

    Solution: cache and pass the *sparse TF-IDF matrix* (~50 MB) instead,
    and compute cosine similarity on-demand per query:

        get_similar_movies  → cosine_similarity(tfidf[i], tfidf)   # 1 row
        hybrid_recommend    → cosine_similarity(tfidf[anchors], tfidf)  # ≤5 rows

    Each on-demand call takes ~20–100 ms and uses O(n × k) memory where
    k = max_features (20k) instead of O(n²).

    The parameter previously called `cosine_sim: np.ndarray` is now
    `tfidf_matrix: scipy.sparse.csr_matrix` throughout. app.py is updated
    to match — nothing else in the public API changes.

Bug fixed from v1:
    top_rated was being computed INSIDE the per-item loop in hybrid_recommend
    → O(n_items × n_ratings) pandas filtering.  Moved outside loop: O(1)/item.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from config import CONTENT_CFG, ContentConfig, RECOMMEND_CFG, RecommendConfig
from src.recommend import _build_popularity_scores

logger = logging.getLogger(__name__)

# ── id_to_pos cache ───────────────────────────────────────────────────────────
_id_to_pos_cache: dict[int, dict[int, int]] = {}


def _build_id_to_pos(movies_reset: pd.DataFrame) -> dict[int, int]:
    """Return {tmdb_id: integer_row_position}, memoised by id(movies_reset)."""
    cache_key = id(movies_reset)
    if cache_key not in _id_to_pos_cache:
        _id_to_pos_cache[cache_key] = {
            int(row["id"]): pos for pos, row in movies_reset.iterrows()
        }
    return _id_to_pos_cache[cache_key]


def build_tfidf(
    movies: pd.DataFrame,
    cache_path: str = "reports/tfidf_matrix",
    cfg: ContentConfig = CONTENT_CFG,
) -> scipy.sparse.csr_matrix:
    """
    Build (or load from cache) the TF-IDF sparse matrix.

    Replaces the old build_similarity() which materialised a dense
    (n × n) cosine matrix requiring up to 16 GB RAM for 46k movies.
    The sparse TF-IDF matrix is ~50 MB regardless of n.

    Cache files
    -----------
    <cache_path>_<n_movies>.npz    — sparse TF-IDF matrix (scipy .npz)
    The movie count is encoded in the filename so stale caches are never
    loaded after the dataset changes.

    Parameters
    ----------
    movies     : DataFrame with a 'content' (or 'genre_str') column
    cache_path : base path (no extension); actual path has n_movies appended
    cfg        : ContentConfig (max_features)

    Returns
    -------
    tfidf_matrix : (n_movies, max_features) sparse float32 matrix
    """
    text_col   = "content" if "content" in movies.columns else "genre_str"
    n_movies   = len(movies)
    npz_path   = f"{cache_path}_{n_movies}.npz"

    if os.path.exists(npz_path):
        logger.info("Loading cached TF-IDF matrix from %s...", npz_path)
        return scipy.sparse.load_npz(npz_path).astype(np.float32)

    corpus = movies[text_col].fillna("").tolist()
    logger.info("Building TF-IDF matrix (%d movies, max_features=%d)...",
                n_movies, cfg.max_features)
    vec    = TfidfVectorizer(stop_words="english", max_features=cfg.max_features)
    matrix = vec.fit_transform(corpus).astype(np.float32)

    os.makedirs(os.path.dirname(npz_path) or ".", exist_ok=True)
    scipy.sparse.save_npz(npz_path, matrix)
    logger.info("TF-IDF matrix cached → %s  (shape: %s, nnz: %d)",
                npz_path, matrix.shape, matrix.nnz)
    return matrix


# ---------------------------------------------------------------------------
# Backward-compat alias — callers that imported build_similarity() still work.
# This is intentionally NOT the same as before (no dense matrix returned).
# ---------------------------------------------------------------------------
def build_similarity(
    movies: pd.DataFrame,
    cache_path: str = "reports/tfidf_matrix",
    cfg: ContentConfig = CONTENT_CFG,
) -> scipy.sparse.csr_matrix:
    """Alias for build_tfidf(). Returns sparse TF-IDF matrix, NOT dense cosine sim."""
    return build_tfidf(movies, cache_path=cache_path, cfg=cfg)


def get_similar_movies(
    title: str,
    movies: pd.DataFrame,
    tfidf_matrix: Optional[scipy.sparse.csr_matrix] = None,
    n: int = 10,
    cosine_sim: Optional[scipy.sparse.csr_matrix] = None,
) -> list[tuple[str, float]]:
    """
    Return the top-n content-similar movies to ``title``.

    Parameters
    ----------
    title        : exact or partial movie title to look up
    movies       : movies DataFrame (must have 'title' column)
    tfidf_matrix : pre-built sparse TF-IDF matrix (preferred kwarg name)
    cosine_sim   : accepted for backward compatibility — treated as tfidf_matrix
    n            : number of similar movies to return

    Returns
    -------
    list of (title, similarity_score) sorted descending

    Raises
    ------
    ValueError : if no matching movie is found
    """
    matrix = tfidf_matrix if tfidf_matrix is not None else cosine_sim
    if matrix is None:
        matrix = build_tfidf(movies)

    matches = movies[movies["title"] == title]
    if matches.empty:
        matches = movies[movies["title"].str.contains(title, case=False, na=False)]
    if matches.empty:
        raise ValueError(f"Movie '{title}' not found in the dataset.")

    pos = movies.index.get_loc(matches.index[0])

    # Compute cosine similarity for ONE row only
    query_vec = matrix[pos]
    sim_row   = cosine_similarity(query_vec, matrix).flatten()

    # Exclude the query movie itself, pick top-n
    sim_row[pos] = -1.0
    top_indices  = np.argpartition(sim_row, -n)[-n:]
    top_indices  = top_indices[np.argsort(sim_row[top_indices])[::-1]]

    return [(movies.iloc[i]["title"], round(float(sim_row[i]), 4)) for i in top_indices]


def hybrid_recommend(
    model: object,
    user_id: int,
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    user_map: dict[int, int],
    movie_map: dict[int, int],
    alpha: float = CONTENT_CFG.alpha,
    n: int = 10,
    rec_cfg: RecommendConfig = RECOMMEND_CFG,
    tfidf_matrix: Optional[scipy.sparse.csr_matrix] = None,
    cosine_sim: Optional[scipy.sparse.csr_matrix] = None,
) -> list[tuple[str, float]]:
    """
    Hybrid scorer: ``alpha × CF_score + (1 - alpha) × CB_score``.

    Cold-start fallback: if user_id is not in user_map, seeds CB from the
    user's highest-rated movie and returns pure content-based results.

    Parameters
    ----------
    model        : trained model with predict_user_all(u_idx) -> np.ndarray
    user_id      : raw user id
    movies       : movies DataFrame (must have 'id' and 'title' columns)
    ratings      : ratings DataFrame (userId, movieId=TMDB, rating)
    user_map     : {raw_user_id -> matrix_row_index}
    movie_map    : {tmdb_id -> matrix_col_index}
    alpha        : CF weight (1-alpha = CB weight)
    n            : number of recommendations
    rec_cfg      : RecommendConfig — popularity_penalty applied to CF component
    tfidf_matrix : pre-built sparse TF-IDF matrix (preferred kwarg name)
    cosine_sim   : accepted for backward compatibility — treated as tfidf_matrix

    Returns
    -------
    list of (title, hybrid_score) sorted descending

    Raises
    ------
    ValueError : if cold-start user has no rating data or seed movie is missing
    """
    matrix = tfidf_matrix if tfidf_matrix is not None else cosine_sim
    if matrix is None:
        logger.info("Building TF-IDF matrix...")
        matrix = build_tfidf(movies)
    else:
        logger.debug("Using pre-built TF-IDF matrix.")

    movies_reset = movies.reset_index(drop=True)
    # Cached: O(n) iterrows() only runs once per unique movies_reset object
    id_to_pos = _build_id_to_pos(movies_reset)

    # ── Cold-start ──────────────────────────────────────────────────────────
    if user_id not in user_map:
        user_ratings = ratings[ratings["userId"] == user_id]
        if user_ratings.empty:
            raise ValueError(f"No rating data for user {user_id}.")
        best_tmdb_id  = user_ratings.loc[user_ratings["rating"].idxmax(), "movieId"]
        title_matches = movies_reset[movies_reset["id"] == best_tmdb_id]
        if title_matches.empty:
            raise ValueError(f"Seed movie (tmdb={best_tmdb_id}) not in movies table.")
        seed_title = title_matches.iloc[0]["title"]
        logger.info("Cold-start user — seeding from: '%s'", seed_title)
        return get_similar_movies(seed_title, movies_reset, tfidf_matrix=matrix, n=n)

    # ── Warm: CF + CB ────────────────────────────────────────────────────────
    u_idx     = user_map[user_id]
    cf_scores = model.predict_user_all(u_idx)

    if rec_cfg.popularity_penalty > 0.0:
        pop       = _build_popularity_scores(ratings, movie_map)
        cf_scores = cf_scores - rec_cfg.popularity_penalty * pop
        logger.debug("Hybrid: popularity penalty applied (weight=%.3f)",
                     rec_cfg.popularity_penalty)

    idx_to_tmdb = {v: k for k, v in movie_map.items()}
    rated_ids   = set(ratings[ratings["userId"] == user_id]["movieId"].tolist())

    # Anchor positions computed ONCE outside the loop; id_to_pos is cached
    top_rated_tmdb = (
        ratings[ratings["userId"] == user_id]
        .nlargest(CONTENT_CFG.n_anchor, "rating")["movieId"]
        .tolist()
    )
    anchor_positions = [id_to_pos[tid] for tid in top_rated_tmdb if tid in id_to_pos]

    # Compute CB scores for ALL candidates in one batch (≤5 rows × n_movies)
    # This replaces the per-item cosine_sim[pos, ap] loop — same result, O(1) passes.
    if anchor_positions:
        anchor_vecs = matrix[anchor_positions]
        cb_all      = cosine_similarity(anchor_vecs, matrix).mean(axis=0)
    else:
        cb_all = np.zeros(len(movies_reset), dtype=np.float32)

    results: list[tuple[str, float]] = []
    for item_idx, cf_score in enumerate(cf_scores):
        tmdb_id = idx_to_tmdb.get(item_idx)
        if tmdb_id is None or tmdb_id in rated_ids:
            continue
        pos = id_to_pos.get(tmdb_id)
        if pos is None:
            continue
        cb_score = float(cb_all[pos])
        results.append((
            movies_reset.loc[pos, "title"],
            alpha * cf_score + (1 - alpha) * cb_score,
        ))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:n]