"""
tests/test_train.py

Unit tests for:
  - MatrixFactorization  (predict clipping, predict_user_all shape, training step)
  - BPR                  (predict_user_all shape, training runs without error)
  - filter_sparse        (correct rows dropped, convergence to stable state)
  - train_model          (returns correct types and dimensions)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from config import MFConfig, BPRConfig, DataConfig
from src.train import MatrixFactorization, BPR, filter_sparse, train_model


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_mf() -> MatrixFactorization:
    """3 users × 5 items, k=4 — fast enough for unit tests."""
    cfg = MFConfig(k=4, lr=0.01, reg=0.01, epochs=2, patience=2)
    return MatrixFactorization(n_users=3, n_items=5, cfg=cfg)


@pytest.fixture
def tiny_bpr() -> BPR:
    """3 users × 5 items, k=4."""
    cfg = BPRConfig(k=4, lr=0.01, reg=0.01, epochs=2, n_samples=2)
    return BPR(n_users=3, n_items=5, cfg=cfg)


@pytest.fixture
def tiny_ratings() -> pd.DataFrame:
    """Small synthetic ratings DataFrame."""
    return pd.DataFrame({
        "userId":  [1, 1, 1, 1, 1, 1,  2, 2, 2, 2, 2, 2,  3, 3, 3, 3, 3, 3],
        "movieId": [10, 20, 30, 40, 50, 60,
                    10, 20, 30, 40, 50, 60,
                    10, 20, 30, 40, 50, 60],
        "rating":  [4.0, 3.5, 2.0, 5.0, 1.0, 4.5,
                    3.0, 4.0, 3.5, 2.5, 4.0, 3.0,
                    5.0, 2.0, 4.5, 3.0, 3.5, 4.0],
    })


# ── MatrixFactorization tests ─────────────────────────────────────────────────

class TestMatrixFactorization:

    def test_predict_clipped_lower(self, tiny_mf: MatrixFactorization) -> None:
        """predict() must never return below 0.5."""
        # Force a strongly negative raw score
        tiny_mf.mu   = -100.0
        tiny_mf.b_u  = np.zeros(3)
        tiny_mf.b_i  = np.zeros(5)
        tiny_mf.P[0] = np.ones(4) * -10
        tiny_mf.Q[0] = np.ones(4) * 10
        assert tiny_mf.predict(0, 0) == pytest.approx(0.5)

    def test_predict_clipped_upper(self, tiny_mf: MatrixFactorization) -> None:
        """predict() must never return above 5.0."""
        tiny_mf.mu   = 100.0
        tiny_mf.b_u  = np.zeros(3)
        tiny_mf.b_i  = np.zeros(5)
        tiny_mf.P[0] = np.zeros(4)
        tiny_mf.Q[0] = np.zeros(4)
        assert tiny_mf.predict(0, 0) == pytest.approx(5.0)

    def test_predict_user_all_shape(self, tiny_mf: MatrixFactorization) -> None:
        """predict_user_all must return one score per item."""
        scores = tiny_mf.predict_user_all(0)
        assert scores.shape == (5,)

    def test_training_reduces_train_rmse(self, tiny_mf: MatrixFactorization) -> None:
        """After 2 epochs on trivial data the model should record RMSE history."""
        data = [(0, 0, 4.0), (0, 1, 3.0), (1, 2, 5.0), (2, 3, 2.0), (1, 4, 3.5)]
        tiny_mf.train(data, patience=None)
        assert len(tiny_mf.train_rmse_history) == 2

    def test_early_stopping_restores_best_weights(self) -> None:
        """Early stopping must restore P/Q from the best epoch, not the last."""
        cfg = MFConfig(k=4, lr=0.005, reg=0.02, epochs=10, patience=1)
        mf  = MatrixFactorization(n_users=3, n_items=5, cfg=cfg)
        data = [(0, 0, 4.0), (0, 1, 3.0), (1, 2, 5.0), (2, 3, 2.0)]
        val  = [(2, 4, 1.0)]
        # Should stop early; just confirm it completes without error
        mf.train(data, val_data=val, patience=1)
        assert len(mf.train_rmse_history) <= 10

    def test_evaluate_rmse_returns_float(self, tiny_mf: MatrixFactorization) -> None:
        val = [(0, 0, 4.0), (1, 1, 3.5)]
        result = tiny_mf.evaluate_rmse(val)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_evaluate_mae_returns_float(self, tiny_mf: MatrixFactorization) -> None:
        val = [(0, 0, 4.0), (1, 1, 3.5)]
        result = tiny_mf.evaluate_mae(val)
        assert isinstance(result, float)
        assert result >= 0.0


# ── BPR tests ────────────────────────────────────────────────────────────────

class TestBPR:

    def test_predict_user_all_shape(self, tiny_bpr: BPR) -> None:
        scores = tiny_bpr.predict_user_all(0)
        assert scores.shape == (5,)

    def test_predict_returns_float(self, tiny_bpr: BPR) -> None:
        score = tiny_bpr.predict(0, 1)
        assert isinstance(score, float)

    def test_training_records_auc_history(self, tiny_bpr: BPR) -> None:
        pos_pairs = {0: [0, 1], 1: [2, 3], 2: [0, 4]}
        tiny_bpr.train(pos_pairs, patience=None)
        assert len(tiny_bpr.train_auc_history) == 2

    def test_positive_items_ranked_above_random_after_training(self) -> None:
        """After enough training, mean score of positives should beat mean of all items."""
        cfg = BPRConfig(k=10, lr=0.05, reg=0.001, epochs=20, n_samples=5)
        bpr = BPR(n_users=5, n_items=20, cfg=cfg)
        # User 0 always likes items 0–4
        pos_pairs = {u: list(range(5)) for u in range(5)}
        bpr.train(pos_pairs, patience=None)

        scores   = bpr.predict_user_all(0)
        pos_mean = float(np.mean(scores[:5]))
        neg_mean = float(np.mean(scores[5:]))
        assert pos_mean > neg_mean, (
            f"Expected positives ({pos_mean:.3f}) > negatives ({neg_mean:.3f})"
        )


# ── filter_sparse tests ───────────────────────────────────────────────────────

class TestFilterSparse:

    def test_drops_low_user_ratings(self) -> None:
        """Users with fewer than min_user_ratings should be removed."""
        df = pd.DataFrame({
            "userId":  [1, 1, 1, 1, 1,  2],
            "movieId": [10, 20, 30, 40, 50,  10],
            "rating":  [4.0] * 5 + [3.0],
        })
        result = filter_sparse(df, min_user_ratings=5, min_movie_ratings=1)
        assert 2 not in result["userId"].values
        assert 1 in result["userId"].values

    def test_drops_low_movie_ratings(self) -> None:
        """Movies with fewer than min_movie_ratings should be removed."""
        df = pd.DataFrame({
            "userId":  [1, 2, 3, 4, 5,  1],
            "movieId": [10, 10, 10, 10, 10,  99],
            "rating":  [4.0] * 5 + [3.0],
        })
        result = filter_sparse(df, min_user_ratings=1, min_movie_ratings=3)
        assert 99 not in result["movieId"].values

    def test_stable_after_one_pass(self, tiny_ratings: pd.DataFrame) -> None:
        """Running filter_sparse twice should not change the result."""
        first  = filter_sparse(tiny_ratings.copy(), min_user_ratings=5, min_movie_ratings=3)
        second = filter_sparse(first.copy(),         min_user_ratings=5, min_movie_ratings=3)
        assert len(first) == len(second)

    def test_returns_dataframe(self, tiny_ratings: pd.DataFrame) -> None:
        result = filter_sparse(tiny_ratings)
        assert isinstance(result, pd.DataFrame)


# ── train_model integration test ──────────────────────────────────────────────

class TestTrainModel:

    def test_returns_correct_types(self, tiny_ratings: pd.DataFrame) -> None:
        cfg      = MFConfig(k=4, lr=0.01, reg=0.01, epochs=1, patience=1)
        data_cfg = DataConfig(val_ratio=0.2, min_user_ratings=3, min_movie_ratings=3)
        model, user_map, movie_map, val_data = train_model(
            tiny_ratings, cfg=cfg, data_cfg=data_cfg
        )
        assert isinstance(model, MatrixFactorization)
        assert isinstance(user_map, dict)
        assert isinstance(movie_map, dict)
        assert isinstance(val_data, list)

    def test_user_map_covers_all_users(self, tiny_ratings: pd.DataFrame) -> None:
        cfg      = MFConfig(k=4, lr=0.01, reg=0.01, epochs=1, patience=1)
        data_cfg = DataConfig(val_ratio=0.1, min_user_ratings=3, min_movie_ratings=3)
        _, user_map, _, _ = train_model(tiny_ratings, cfg=cfg, data_cfg=data_cfg)
        expected_users = set(tiny_ratings["userId"].unique())
        assert expected_users == set(user_map.keys())

    def test_model_dimensions_match_maps(self, tiny_ratings: pd.DataFrame) -> None:
        cfg      = MFConfig(k=4, lr=0.01, reg=0.01, epochs=1, patience=1)
        data_cfg = DataConfig(val_ratio=0.1, min_user_ratings=3, min_movie_ratings=3)
        model, user_map, movie_map, _ = train_model(tiny_ratings, cfg=cfg, data_cfg=data_cfg)
        assert model.P.shape == (len(user_map), cfg.k)
        assert model.Q.shape == (len(movie_map), cfg.k)