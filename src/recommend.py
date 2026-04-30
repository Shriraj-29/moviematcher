"""
recommend.py — Vectorised CF top-N recommendation.

Key notes
---------
- movies['id']      = TMDB id  (NOT MovieLens movieId)
- movie_map keys    = TMDB ids (ints), mapped in data_loader
- ratings.movieId   = TMDB id  (remapped in data_loader)

All scoring is done in one vectorised call (P[u] @ Q.T), so no Python
loops over items — O(1) per recommendation call after the dot product.

Coverage fix
------------
Pure CF suffers from popularity bias: the model sees blockbusters in
almost every training row, so their latent vectors are pulled toward
higher scores for most users.  The result is that only ~2% of the
catalog ever appears in top-N lists.

We apply a log-normalised popularity penalty *after* the dot product,
before sorting.  Because it is subtracted uniformly from every
candidate's score, relative ordering among equally-popular films is
preserved — only the systematic advantage of high-count films shrinks.

    adjusted = cf_score - penalty * log(count+1) / log(max_count+1)

penalty=0.0 reproduces the original behaviour exactly.

Caching
-------
_build_popularity_scores() is memoised by (id(ratings), id(movie_map)) so
the O(n_ratings) value_counts() + O(n_items) loop runs only once per
unique (ratings, movie_map) pair — not on every recommendation request.

_build_id_to_title() is memoised by id(movies) for the same reason:
movies.set_index().to_dict() scans the whole frame every call.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import RECOMMEND_CFG, RecommendConfig

logger = logging.getLogger(__name__)


# ── Popularity penalty helper ─────────────────────────────────────────────────

# Cache: (id(ratings), id(movie_map)) -> np.ndarray
_popularity_cache: dict[tuple[int, int], np.ndarray] = {}


def _build_popularity_scores(
    ratings: pd.DataFrame,
    movie_map: dict[int, int],
) -> np.ndarray:
    """
    Return a (n_items,) array of log-normalised popularity scores in [0, 1].

    popularity[i] = log(rating_count_for_item_i + 1) / log(max_count + 1)

    Items with no ratings in the DataFrame get a score of 0.0 (no penalty).
    The array is indexed by *matrix column index*, matching predict_user_all().

    Result is memoised by (id(ratings), id(movie_map)) so repeated calls
    with the same objects are free after the first computation.
    """
    cache_key = (id(ratings), id(movie_map))
    if cache_key in _popularity_cache:
        return _popularity_cache[cache_key]

    counts  = ratings["movieId"].value_counts()
    n_items = len(movie_map)

    # Vectorised fill via reindex — no Python loop over movie_map
    tmdb_ids   = np.array(list(movie_map.keys()),   dtype=np.int64)
    col_idxs   = np.array(list(movie_map.values()), dtype=np.int64)
    raw_counts = counts.reindex(tmdb_ids, fill_value=0).to_numpy(dtype=np.float32)

    raw = np.zeros(n_items, dtype=np.float32)
    raw[col_idxs] = raw_counts

    max_count = raw.max()
    result = (
        np.zeros(n_items, dtype=np.float32) if max_count <= 0
        else np.log1p(raw) / np.log1p(max_count)
    )

    _popularity_cache[cache_key] = result
    return result


# ── id_to_title helper ────────────────────────────────────────────────────────

# Cache: id(movies) -> dict[int, str]
_id_to_title_cache: dict[int, dict[int, str]] = {}


def _build_id_to_title(movies: pd.DataFrame) -> dict[int, str]:
    """
    Return {tmdb_id: title} dict, memoised by id(movies).

    movies.set_index("id")["title"].to_dict() scans the full DataFrame on
    every call; caching avoids that cost on repeated get_top_n() requests.
    """
    cache_key = id(movies)
    if cache_key not in _id_to_title_cache:
        _id_to_title_cache[cache_key] = movies.set_index("id")["title"].to_dict()
    return _id_to_title_cache[cache_key]


# ── Public API ────────────────────────────────────────────────────────────────

def get_top_n(
    model: object,
    user_id: int,
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    user_map: dict[int, int],
    movie_map: dict[int, int],
    n: int = 10,
    cfg: RecommendConfig = RECOMMEND_CFG,
) -> list[tuple[str, float]]:
    """
    Return top-n unseen movies for user_id ranked by penalised CF score.

    Parameters
    ----------
    model     : trained model with predict_user_all(u_idx) -> np.ndarray
    user_id   : raw user id (must exist in user_map)
    movies    : DataFrame with 'id' (TMDB) and 'title' columns
    ratings   : DataFrame with 'userId' and 'movieId' (TMDB) columns
    user_map  : {raw_user_id -> matrix_row_index}
    movie_map : {tmdb_id -> matrix_col_index}
    n         : number of recommendations to return
    cfg       : RecommendConfig -- controls popularity_penalty weight

    Returns
    -------
    list of (title, adjusted_score) sorted descending

    Raises
    ------
    ValueError : if user_id is not in training data (cold-start)
    """
    if user_id not in user_map:
        raise ValueError(
            f"User {user_id} not in training data. "
            "Use hybrid_recommend() from content_based.py for cold-start users."
        )

    u_idx = user_map[user_id]
    logger.debug("Scoring all items for user %d (matrix row %d)...", user_id, u_idx)

    # Items the user has already rated -- exclude from recommendations
    rated_tmdb_ids = set(ratings[ratings["userId"] == user_id]["movieId"].tolist())
    rated_indices  = {movie_map[mid] for mid in rated_tmdb_ids if mid in movie_map}

    # ── Vectorised CF scores ───────────────────────────────────────────────
    all_scores = model.predict_user_all(u_idx)

    # ── Popularity penalty (vectorised; cached after first call) ───────────
    if cfg.popularity_penalty > 0.0:
        pop        = _build_popularity_scores(ratings, movie_map)
        all_scores = all_scores - cfg.popularity_penalty * pop
        logger.debug(
            "Popularity penalty applied (weight=%.3f); "
            "score range: [%.3f, %.3f]",
            cfg.popularity_penalty, float(all_scores.min()), float(all_scores.max()),
        )

    # ── Filter rated items and pick top-N ─────────────────────────────────
    candidates = [
        (tmdb_id, float(all_scores[idx]))
        for tmdb_id, idx in movie_map.items()
        if idx not in rated_indices
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)
    top = candidates[:n]

    # Cached — free after first call per unique movies DataFrame object
    id_to_title = _build_id_to_title(movies)
    results = [
        (id_to_title.get(tmdb_id, f"Unknown(tmdb={tmdb_id})"), score)
        for tmdb_id, score in top
    ]
    logger.debug("Top-%d recommendations generated for user %d.", n, user_id)
    return results