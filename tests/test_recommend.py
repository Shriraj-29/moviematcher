"""
tests/test_recommend.py

Unit tests for:
  - get_top_n         (filters rated items, correct length, raises on unknown user)
  - get_similar_movies (correct length, raises on unknown title)
  - hybrid_recommend  (cold-start path, warm path returns n results)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from config import MFConfig
from src.train import MatrixFactorization
from src.recommend import get_top_n
from src.content_based import get_similar_movies, hybrid_recommend


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def small_movies() -> pd.DataFrame:
    return pd.DataFrame({
        "id":        [10, 20, 30, 40, 50],
        "title":     ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
        "genre_str": ["Action", "Drama", "Comedy Action", "Drama Romance", "Thriller"],
        "content":   ["Action hero film", "Drama sad story", "Comedy action laugh",
                      "Drama romance love", "Thriller suspense"],
    })


@pytest.fixture
def small_ratings() -> pd.DataFrame:
    return pd.DataFrame({
        "userId":  [1, 1, 1,   2, 2,   3, 3, 3],
        "movieId": [10, 20, 30, 10, 40, 10, 20, 50],
        "rating":  [5.0, 3.0, 4.0, 2.0, 4.5, 3.5, 4.0, 5.0],
    })


@pytest.fixture
def small_maps() -> tuple[dict, dict]:
    """user_map and movie_map for the fixtures above."""
    user_map  = {1: 0, 2: 1, 3: 2}
    movie_map = {10: 0, 20: 1, 30: 2, 40: 3, 50: 4}
    return user_map, movie_map


@pytest.fixture
def trained_mf(small_maps: tuple[dict, dict]) -> MatrixFactorization:
    """Pre-built MF with deterministic weights (no training needed for routing tests)."""
    user_map, movie_map = small_maps
    cfg = MFConfig(k=4, lr=0.01, reg=0.01, epochs=1)
    mf  = MatrixFactorization(n_users=len(user_map), n_items=len(movie_map), cfg=cfg)
    mf.mu = 3.5
    return mf


# ── get_top_n tests ───────────────────────────────────────────────────────────

class TestGetTopN:

    def test_raises_for_unknown_user(
        self,
        trained_mf: MatrixFactorization,
        small_movies: pd.DataFrame,
        small_ratings: pd.DataFrame,
        small_maps: tuple[dict, dict],
    ) -> None:
        user_map, movie_map = small_maps
        with pytest.raises(ValueError, match="not in training data"):
            get_top_n(trained_mf, user_id=999, movies=small_movies,
                      ratings=small_ratings, user_map=user_map, movie_map=movie_map)

    def test_excludes_already_rated_items(
        self,
        trained_mf: MatrixFactorization,
        small_movies: pd.DataFrame,
        small_ratings: pd.DataFrame,
        small_maps: tuple[dict, dict],
    ) -> None:
        user_map, movie_map = small_maps
        # User 1 rated movies 10, 20, 30 → only 40, 50 should be returned
        recs = get_top_n(trained_mf, user_id=1, movies=small_movies,
                         ratings=small_ratings, user_map=user_map, movie_map=movie_map)
        returned_titles = {title for title, _ in recs}
        assert "Alpha" not in returned_titles   # movie 10 — rated
        assert "Beta"  not in returned_titles   # movie 20 — rated
        assert "Gamma" not in returned_titles   # movie 30 — rated

    def test_returns_at_most_n_items(
        self,
        trained_mf: MatrixFactorization,
        small_movies: pd.DataFrame,
        small_ratings: pd.DataFrame,
        small_maps: tuple[dict, dict],
    ) -> None:
        user_map, movie_map = small_maps
        for n in [1, 2]:
            recs = get_top_n(trained_mf, user_id=1, movies=small_movies,
                             ratings=small_ratings, user_map=user_map,
                             movie_map=movie_map, n=n)
            assert len(recs) <= n

    def test_scores_sorted_descending(
        self,
        trained_mf: MatrixFactorization,
        small_movies: pd.DataFrame,
        small_ratings: pd.DataFrame,
        small_maps: tuple[dict, dict],
    ) -> None:
        user_map, movie_map = small_maps
        # User 2 has rated 10, 40 → 3 unseen items
        recs = get_top_n(trained_mf, user_id=2, movies=small_movies,
                         ratings=small_ratings, user_map=user_map,
                         movie_map=movie_map, n=3)
        scores = [s for _, s in recs]
        assert scores == sorted(scores, reverse=True)

    def test_returns_list_of_tuples(
        self,
        trained_mf: MatrixFactorization,
        small_movies: pd.DataFrame,
        small_ratings: pd.DataFrame,
        small_maps: tuple[dict, dict],
    ) -> None:
        user_map, movie_map = small_maps
        recs = get_top_n(trained_mf, user_id=2, movies=small_movies,
                         ratings=small_ratings, user_map=user_map, movie_map=movie_map)
        assert isinstance(recs, list)
        for item in recs:
            assert isinstance(item, tuple) and len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], float)


# ── get_similar_movies tests ──────────────────────────────────────────────────

class TestGetSimilarMovies:

    def test_raises_for_unknown_title(self, small_movies: pd.DataFrame) -> None:
        sim = np.eye(len(small_movies))
        with pytest.raises(ValueError, match="not found"):
            get_similar_movies("Nonexistent Movie", small_movies, cosine_sim=sim)

    def test_returns_n_results(self, small_movies: pd.DataFrame) -> None:
        # Build a real sim matrix from the fixture
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as cs
        vec = TfidfVectorizer()
        mat = vec.fit_transform(small_movies["content"].fillna(""))
        sim = cs(mat, mat)
        recs = get_similar_movies("Alpha", small_movies, cosine_sim=sim, n=2)
        assert len(recs) == 2

    def test_does_not_include_query_movie(self, small_movies: pd.DataFrame) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as cs
        vec = TfidfVectorizer()
        mat = vec.fit_transform(small_movies["content"].fillna(""))
        sim = cs(mat, mat)
        recs = get_similar_movies("Alpha", small_movies, cosine_sim=sim, n=4)
        titles = [t for t, _ in recs]
        assert "Alpha" not in titles

    def test_partial_title_match(self, small_movies: pd.DataFrame) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as cs
        vec = TfidfVectorizer()
        mat = vec.fit_transform(small_movies["content"].fillna(""))
        sim = cs(mat, mat)
        recs = get_similar_movies("lph", small_movies, cosine_sim=sim, n=2)
        assert len(recs) == 2


# ── hybrid_recommend tests ────────────────────────────────────────────────────

class TestHybridRecommend:

    def test_cold_start_raises_if_no_ratings(
        self,
        trained_mf: MatrixFactorization,
        small_movies: pd.DataFrame,
        small_ratings: pd.DataFrame,
        small_maps: tuple[dict, dict],
    ) -> None:
        user_map, movie_map = small_maps
        empty_ratings = pd.DataFrame({"userId": [], "movieId": [], "rating": []})
        with pytest.raises(ValueError, match="No rating data"):
            hybrid_recommend(
                trained_mf, user_id=999,
                movies=small_movies, ratings=empty_ratings,
                user_map=user_map, movie_map=movie_map, n=2,
            )

    def test_warm_returns_n_results(
        self,
        trained_mf: MatrixFactorization,
        small_movies: pd.DataFrame,
        small_ratings: pd.DataFrame,
        small_maps: tuple[dict, dict],
    ) -> None:
        user_map, movie_map = small_maps
        recs = hybrid_recommend(
            trained_mf, user_id=1,
            movies=small_movies, ratings=small_ratings,
            user_map=user_map, movie_map=movie_map, n=2,
        )
        assert len(recs) <= 2

    def test_warm_excludes_rated_items(
        self,
        trained_mf: MatrixFactorization,
        small_movies: pd.DataFrame,
        small_ratings: pd.DataFrame,
        small_maps: tuple[dict, dict],
    ) -> None:
        user_map, movie_map = small_maps
        # User 1 rated 10 (Alpha), 20 (Beta), 30 (Gamma)
        recs = hybrid_recommend(
            trained_mf, user_id=1,
            movies=small_movies, ratings=small_ratings,
            user_map=user_map, movie_map=movie_map, n=5,
        )
        titles = {t for t, _ in recs}
        for rated in ["Alpha", "Beta", "Gamma"]:
            assert rated not in titles