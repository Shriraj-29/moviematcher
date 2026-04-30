"""
tests/test_evaluate.py

Unit tests for:
  - rmse / mae                  (correct computation, perfect predictions → 0)
  - precision_recall_at_k       (edge cases: no relevant items, perfect ranker)
  - catalog_coverage            (boundary values)
  - intra_list_diversity        (same-genre → 0, different-genre → > 0)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    rmse, mae,
    precision_recall_at_k,
    catalog_coverage,
    intra_list_diversity,
)


# ── Stub model ────────────────────────────────────────────────────────────────

class _ConstantModel:
    """Always predicts the same value — useful for deterministic metric tests."""
    def __init__(self, value: float) -> None:
        self._value = value

    def predict(self, u: int, i: int) -> float:
        return self._value

    def predict_user_all(self, u: int) -> np.ndarray:
        return np.full(10, self._value)


class _PerfectModel:
    """Returns the exact rating passed in a lookup dict."""
    def __init__(self, lookup: dict) -> None:
        self._lookup = lookup

    def predict(self, u: int, i: int) -> float:
        return self._lookup.get((u, i), 3.0)


# ── rmse / mae ────────────────────────────────────────────────────────────────

class TestRmseAndMae:

    def test_rmse_perfect_predictions(self) -> None:
        val = [(0, 0, 4.0), (1, 1, 3.5), (0, 2, 2.0)]
        model = _PerfectModel({(0, 0): 4.0, (1, 1): 3.5, (0, 2): 2.0})
        assert rmse(model, val) == pytest.approx(0.0, abs=1e-6)

    def test_mae_perfect_predictions(self) -> None:
        val = [(0, 0, 4.0), (1, 1, 3.0)]
        model = _PerfectModel({(0, 0): 4.0, (1, 1): 3.0})
        assert mae(model, val) == pytest.approx(0.0, abs=1e-6)

    def test_rmse_known_value(self) -> None:
        """Constant model predicting 3.0 on ratings [5.0, 1.0] → RMSE = 2.0."""
        val   = [(0, 0, 5.0), (1, 1, 1.0)]
        model = _ConstantModel(3.0)
        assert rmse(model, val) == pytest.approx(2.0, abs=1e-4)

    def test_mae_known_value(self) -> None:
        """Constant model predicting 3.0 on ratings [5.0, 1.0] → MAE = 2.0."""
        val   = [(0, 0, 5.0), (1, 1, 1.0)]
        model = _ConstantModel(3.0)
        assert mae(model, val) == pytest.approx(2.0, abs=1e-4)

    def test_rmse_non_negative(self) -> None:
        val   = [(0, 0, 4.0), (1, 1, 2.5), (0, 2, 3.0)]
        model = _ConstantModel(3.5)
        assert rmse(model, val) >= 0.0

    def test_mae_non_negative(self) -> None:
        val   = [(0, 0, 4.0), (1, 1, 2.5)]
        model = _ConstantModel(4.0)
        assert mae(model, val) >= 0.0


# ── precision_recall_at_k ─────────────────────────────────────────────────────

class TestPrecisionRecallAtK:

    def _make_val(self) -> list[tuple]:
        # User 0: items 0(5.0 ✓), 1(4.0 ✓), 2(2.0 ✗), 3(1.5 ✗)
        # User 1: items 4(5.0 ✓), 5(1.0 ✗)
        return [
            (0, 0, 5.0), (0, 1, 4.0), (0, 2, 2.0), (0, 3, 1.5),
            (1, 4, 5.0), (1, 5, 1.0),
        ]

    def test_perfect_ranker_precision_1(self) -> None:
        """A ranker that scores relevant items highest should achieve P@2 = 1.0."""
        val = self._make_val()
        # User 0 relevant = {0,1}; User 1 relevant = {4}
        lookup = {
            (0, 0): 5.0, (0, 1): 4.0, (0, 2): 1.0, (0, 3): 0.5,
            (1, 4): 5.0, (1, 5): 0.5,
        }
        model = _PerfectModel(lookup)
        p, _ = precision_recall_at_k(model, val, k=2, threshold=3.5)
        assert p == pytest.approx(1.0, abs=0.01)

    def test_recall_at_k_with_perfect_ranker(self) -> None:
        val    = self._make_val()
        lookup = {(0,0):5.0,(0,1):4.0,(0,2):1.0,(0,3):0.5,(1,4):5.0,(1,5):0.5}
        model  = _PerfectModel(lookup)
        _, r   = precision_recall_at_k(model, val, k=2, threshold=3.5)
        # User 0: hits=2 / relevant=2 = 1.0;  User 1: hits=1 / relevant=1 = 1.0
        assert r == pytest.approx(1.0, abs=0.01)

    def test_returns_floats(self) -> None:
        val   = [(0, 0, 4.0), (0, 1, 2.0)]
        model = _ConstantModel(3.0)
        p, r  = precision_recall_at_k(model, val, k=1)
        assert isinstance(p, float) and isinstance(r, float)

    def test_no_relevant_items_skipped_gracefully(self) -> None:
        """Users with no relevant items should be skipped without error."""
        val   = [(0, 0, 1.0), (0, 1, 2.0)]
        model = _ConstantModel(3.0)
        p, r  = precision_recall_at_k(model, val, k=2, threshold=3.5)
        assert isinstance(p, float)


# ── catalog_coverage ─────────────────────────────────────────────────────────

class TestCatalogCoverage:

    def test_full_coverage(self) -> None:
        assert catalog_coverage({1, 2, 3, 4, 5}, total_items=5) == pytest.approx(1.0)

    def test_zero_coverage(self) -> None:
        assert catalog_coverage(set(), total_items=100) == pytest.approx(0.0)

    def test_partial_coverage(self) -> None:
        assert catalog_coverage({1, 2}, total_items=4) == pytest.approx(0.5)

    def test_coverage_is_between_0_and_1(self) -> None:
        cov = catalog_coverage({1, 2, 3}, total_items=10)
        assert 0.0 <= cov <= 1.0


# ── intra_list_diversity ──────────────────────────────────────────────────────

class TestIntraListDiversity:

    @pytest.fixture
    def genre_movies(self) -> pd.DataFrame:
        return pd.DataFrame({
            "id":        [1, 2, 3, 4],
            "genre_str": ["Action", "Action", "Drama", "Comedy"],
        })

    def test_same_genre_low_diversity(self, genre_movies: pd.DataFrame) -> None:
        """All-Action list should have near-zero ILD."""
        recs = [[1, 2]]
        div  = intra_list_diversity(recs, genre_movies)
        assert div == pytest.approx(0.0, abs=0.05)

    def test_different_genres_higher_diversity(self, genre_movies: pd.DataFrame) -> None:
        """Mixed-genre list should have higher ILD than same-genre."""
        same_genre   = [[1, 2]]
        mixed_genre  = [[1, 3]]
        div_same  = intra_list_diversity(same_genre,  genre_movies)
        div_mixed = intra_list_diversity(mixed_genre, genre_movies)
        assert div_mixed > div_same

    def test_single_item_list_skipped(self, genre_movies: pd.DataFrame) -> None:
        """Lists with one item cannot compute pairwise distance — return 0.0."""
        recs = [[1]]
        div  = intra_list_diversity(recs, genre_movies)
        assert div == pytest.approx(0.0, abs=1e-6)

    def test_diversity_between_0_and_1(self, genre_movies: pd.DataFrame) -> None:
        recs = [[1, 3, 4]]
        div  = intra_list_diversity(recs, genre_movies)
        assert 0.0 <= div <= 1.0

    def test_unknown_ids_ignored(self, genre_movies: pd.DataFrame) -> None:
        """IDs not in movies table should be silently skipped."""
        recs = [[1, 999]]
        div  = intra_list_diversity(recs, genre_movies)
        assert isinstance(div, float)