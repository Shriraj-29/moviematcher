"""
config.py — Central configuration for MovieMatcher.

All hyperparameters, paths, and constants live here.
Import the pre-built singletons (MF_CFG, BPR_CFG, etc.) or
override individual fields before passing to train_model().

Example:
    from config import MF_CFG
    MF_CFG.lr = 0.01
"""

from dataclasses import dataclass
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data")
MODELS_DIR  = Path("models")
REPORTS_DIR = Path("reports")

MODEL_PATH  = MODELS_DIR / "mf.pkl"
BPR_PATH    = MODELS_DIR / "bpr.pkl"
METRICS_PATH = REPORTS_DIR / "metrics.txt"


# ── Matrix Factorization ────────────────────────────────────────────────────
@dataclass
class MFConfig:
    """Hyperparameters for SGD-based Matrix Factorization."""
    k: int        = 50       # number of latent factors
    lr: float     = 0.005    # SGD learning rate
    reg: float    = 0.02     # L2 regularisation coefficient
    epochs: int   = 20       # maximum training epochs
    patience: int = 3        # early-stopping patience (epochs without val improvement)


# ── Bayesian Personalised Ranking ───────────────────────────────────────────
@dataclass
class BPRConfig:
    """Hyperparameters for BPR pairwise ranking model."""
    k: int        = 50       # latent factors
    lr: float     = 0.01     # learning rate (BPR benefits from slightly higher lr)
    reg: float    = 0.01     # L2 regularisation
    epochs: int   = 20       # max epochs
    patience: int = 3        # early-stopping patience
    n_samples: int = 5       # negative samples drawn per positive interaction


# ── Data ────────────────────────────────────────────────────────────────────
@dataclass
class DataConfig:
    """Train/val split and sparsity-filter thresholds."""
    val_ratio: float          = 0.1   # fraction held out for validation
    min_user_ratings: int     = 5     # drop users with fewer ratings
    min_movie_ratings: int    = 10    # drop movies with fewer ratings


# ── Content-Based ───────────────────────────────────────────────────────────
@dataclass
class ContentConfig:
    """TF-IDF and hybrid scoring settings."""
    max_features: int = 20_000   # TF-IDF vocabulary cap
    alpha: float      = 0.7      # hybrid weight: alpha*CF + (1-alpha)*CB
    n_anchor: int     = 5        # top-rated movies used as CB anchors


# ── Recommendation / Coverage ────────────────────────────────────────────────
@dataclass
class RecommendConfig:
    """
    Post-scoring adjustments applied before final top-N selection.

    popularity_penalty
        Weight applied to a log-normalised popularity score that is
        *subtracted* from every candidate's raw CF (or hybrid) score:

            adjusted = raw_score − penalty × log(count+1) / log(max_count+1)

        0.0  → no adjustment (pure CF behaviour, high popularity bias)
        0.05 → mild nudge toward less-seen films (recommended default)
        0.15 → strong de-emphasis of blockbusters
        ≥0.3 → may hurt precision noticeably — only use if diversity is
                the primary objective

    The penalty is computed per call and adds negligible overhead (one
    vectorised NumPy op over the candidate list).
    """
    popularity_penalty: float = 0.05


# ── Evaluation ──────────────────────────────────────────────────────────────
@dataclass
class EvalConfig:
    """Evaluation metric settings."""
    k: int            = 10     # list length for Precision/Recall@K
    threshold: float  = 3.5   # minimum rating to count as "relevant"
    sample_users: int = 500   # users sampled for coverage/diversity
                               # (raised from 200 — more users → fairer coverage estimate)


# ── Pre-built singletons (import these directly) ────────────────────────────
MF_CFG        = MFConfig()
BPR_CFG       = BPRConfig()
DATA_CFG      = DataConfig()
CONTENT_CFG   = ContentConfig()
EVAL_CFG      = EvalConfig()
RECOMMEND_CFG = RecommendConfig()