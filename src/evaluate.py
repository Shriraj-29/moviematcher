"""
evaluate.py — Recommender system evaluation + chart generation.

Charts saved to reports/:
    rmse_convergence.png     train vs val RMSE across epochs
    metrics_summary.png      bar chart: RMSE, MAE, Precision@K, Recall@K
    coverage_diversity.png   horizontal gauge bars for coverage and diversity
    rating_distribution.png  predicted vs actual rating histogram
"""

from __future__ import annotations

import logging
import os
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from config import EVAL_CFG, EvalConfig

logger = logging.getLogger(__name__)

# ── Chart styling ────────────────────────────────────────────────────────────
BLUE   = "#4C72B0"
ORANGE = "#DD8452"
GREEN  = "#55A868"
RED    = "#C44E52"
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size": 11,
})


# ── Rating accuracy ──────────────────────────────────────────────────────────

def rmse(model: Any, val_data: list[tuple]) -> float:
    """Root-mean-squared error on held-out val triples."""
    errors = [
        (r - model.predict(int(u), int(i))) ** 2
        for u, i, r in tqdm(val_data, desc="  RMSE", unit="sample", leave=False)
    ]
    return float(np.sqrt(np.mean(errors)))


def mae(model: Any, val_data: list[tuple]) -> float:
    """Mean absolute error on held-out val triples."""
    errors = [
        abs(r - model.predict(int(u), int(i)))
        for u, i, r in tqdm(val_data, desc="  MAE ", unit="sample", leave=False)
    ]
    return float(np.mean(errors))


# ── Ranking metrics ──────────────────────────────────────────────────────────

def precision_recall_at_k(
    model: Any,
    val_data: list[tuple],
    k: int = EVAL_CFG.k,
    threshold: float = EVAL_CFG.threshold,
) -> tuple[float, float]:
    """
    Mean Precision@K and Recall@K across all users in val_data.

    Parameters
    ----------
    threshold : minimum rating to count as "relevant"

    Returns
    -------
    (precision_at_k, recall_at_k)
    """
    # Build a DataFrame and group by user — single pass, no Python dict building
    df = pd.DataFrame(val_data, columns=["u", "i", "r"])
    df["u"] = df["u"].astype(int)
    df["i"] = df["i"].astype(int)
    df["r"] = df["r"].astype(float)

    precisions, recalls = [], []
    for u_idx, group in tqdm(
        df.groupby("u", sort=False),
        desc="  Precision/Recall@K", unit="user", leave=False,
    ):
        relevant = set(group.loc[group["r"] >= threshold, "i"])
        if not relevant:
            continue
        scores = sorted(
            [(int(row["i"]), model.predict(int(u_idx), int(row["i"])))
             for _, row in group.iterrows()],
            key=lambda x: x[1], reverse=True,
        )
        top_k = {i for i, _ in scores[:k]}
        hits  = len(top_k & relevant)
        precisions.append(hits / k)
        recalls.append(hits / len(relevant))

    return float(np.mean(precisions)), float(np.mean(recalls))


# ── Coverage ─────────────────────────────────────────────────────────────────

def catalog_coverage(all_recommended_ids: set[int], total_items: int) -> float:
    """Fraction of the catalog surfaced across all recommendation lists."""
    return len(all_recommended_ids) / total_items


# ── Diversity (ILD) ──────────────────────────────────────────────────────────

def intra_list_diversity(
    rec_tmdb_ids_list: list[list[int]],
    movies: pd.DataFrame,
) -> float:
    """
    Average pairwise genre-cosine-distance within each recommendation list.

    1.0 = maximally diverse (all different genres)
    0.0 = no diversity (all same genre)

    Memory-efficient: builds the TF-IDF + similarity matrix only for the
    unique movies that appear in recommendation lists, not the full catalog.
    For 200 users × top-10 this is at most a ~2k × 2k matrix (~30 MB)
    instead of 45k × 45k (~15 GB).
    """
    genre_col = "genre_str" if "genre_str" in movies.columns else "genres"

    # ── 1. Collect only the unique TMDB ids that were actually recommended ──
    unique_ids: list[int] = list({tid for lst in rec_tmdb_ids_list for tid in lst})
    if len(unique_ids) < 2:
        return 0.0

    # ── 2. Subset movies to just those ids ──────────────────────────────────
    subset = (
        movies[movies["id"].isin(unique_ids)]
        .drop_duplicates(subset="id")
        .reset_index(drop=True)
    )
    subset[genre_col] = subset[genre_col].fillna("")

    # ── 3. Build TF-IDF + similarity on the small subset only ───────────────
    vec = CountVectorizer()
    try:
        genre_matrix = vec.fit_transform(subset[genre_col])
    except ValueError:
        return 0.0

    # subset is small (≤ n_users × top_k), so this matrix is tiny
    sim_matrix  = cosine_similarity(genre_matrix)
    tmdb_to_pos = {int(row["id"]): pos for pos, row in subset.iterrows()}

    # ── 4. Compute ILD per list ──────────────────────────────────────────────
    diversities = []
    for rec_ids in tqdm(
        rec_tmdb_ids_list, desc="  Diversity", unit="list", leave=False
    ):
        positions = [tmdb_to_pos[tid] for tid in rec_ids if tid in tmdb_to_pos]
        if len(positions) < 2:
            continue
        dists = [
            1 - sim_matrix[positions[a], positions[b]]
            for a in range(len(positions))
            for b in range(a + 1, len(positions))
        ]
        diversities.append(float(np.mean(dists)))

    return float(np.mean(diversities)) if diversities else 0.0


# ── Charts ───────────────────────────────────────────────────────────────────

def _plot_convergence(model: Any, out_dir: str) -> None:
    if not model.train_rmse_history:
        return
    epochs = range(1, len(model.train_rmse_history) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, model.train_rmse_history, color=BLUE,   marker="o", markersize=4, label="Train RMSE")
    if model.val_rmse_history:
        ax.plot(epochs, model.val_rmse_history, color=ORANGE, marker="s", markersize=4, label="Val RMSE")
    ax.set_xlabel("Epoch"); ax.set_ylabel("RMSE"); ax.set_title("Training Convergence")
    ax.legend(); ax.set_xticks(list(epochs))
    fig.tight_layout()
    fig.savefig(f"{out_dir}/rmse_convergence.png", dpi=150)
    plt.close(fig)
    logger.info("Saved → %s/rmse_convergence.png", out_dir)


def _plot_metrics_summary(metrics: dict[str, float], k: int, out_dir: str) -> None:
    labels = ["RMSE", "MAE", f"Precision@{k}", f"Recall@{k}"]
    values = [metrics["rmse"], metrics["mae"], metrics["precision_at_k"], metrics["recall_at_k"]]
    colors = [RED, ORANGE, BLUE, GREEN]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors, width=0.5, zorder=2)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=1)
    ax.set_ylim(0, max(values) * 1.25)
    ax.set_title("Evaluation Metrics Summary"); ax.set_ylabel("Score")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.4f}", ha="center", va="bottom", fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(f"{out_dir}/metrics_summary.png", dpi=150)
    plt.close(fig)
    logger.info("Saved → %s/metrics_summary.png", out_dir)


def _plot_coverage_diversity(metrics: dict[str, float], out_dir: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7, 3))
    fig.suptitle("Coverage & Diversity", fontsize=13, fontweight="bold", y=1.02)
    pairs = [
        ("Coverage",  metrics["coverage"],  GREEN, "% of catalog recommended"),
        ("Diversity", metrics["diversity"], BLUE,  "Intra-list genre diversity (ILD)"),
    ]
    for ax, (label, value, color, subtitle) in zip(axes, pairs):
        ax.barh([0], [1],     color="#E8E8E8", height=0.5)
        ax.barh([0], [value], color=color,     height=0.5, zorder=2)
        ax.set_xlim(0, 1); ax.set_yticks([])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_title(f"{label}: {value:.3f}  —  {subtitle}", fontsize=10, loc="left")
        ax.spines["left"].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/coverage_diversity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s/coverage_diversity.png", out_dir)


def _plot_rating_distribution(
    model: Any, val_data: list[tuple], out_dir: str
) -> None:
    actuals   = [r for _, _, r in val_data]
    predicted = [
        model.predict(int(u), int(i))
        for u, i, _ in tqdm(val_data, desc="  Rating dist", unit="sample", leave=False)
    ]
    bins = np.arange(0.5, 5.6, 0.5)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(actuals,   bins=bins, alpha=0.6, color=BLUE,   label="Actual",    density=True)
    ax.hist(predicted, bins=bins, alpha=0.6, color=ORANGE, label="Predicted", density=True)
    ax.set_xlabel("Rating"); ax.set_ylabel("Density")
    ax.set_title("Predicted vs Actual Rating Distribution (Val Set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_dir}/rating_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Saved → %s/rating_distribution.png", out_dir)


# ── Full evaluation pipeline ─────────────────────────────────────────────────

def run_evaluation(
    model: Any,
    val_data: list[tuple],
    user_map: dict[int, int],
    movie_map: dict[int, int],
    movies: pd.DataFrame,
    ratings: pd.DataFrame | None = None,
    cfg: EvalConfig = EVAL_CFG,
    rec_cfg: Any = None,
) -> dict[str, float]:
    """
    Compute all metrics, write reports/metrics.txt, and save charts.

    Parameters
    ----------
    model     : trained model with .predict(u, i) and .predict_user_all(u)
    val_data  : held-out (u_idx, i_idx, rating) triples
    user_map  : {raw_user_id -> matrix_row_index}
    movie_map : {tmdb_id -> matrix_col_index}
    movies    : movies DataFrame
    ratings   : full ratings DataFrame (needed to apply popularity penalty in
                coverage scoring; pass None to skip penalty during evaluation)
    cfg       : EvalConfig (k, threshold, sample_users)
    rec_cfg   : RecommendConfig — if provided, applies popularity_penalty when
                scoring users for coverage/diversity so evaluation reflects the
                same score distribution seen at recommendation time

    Returns
    -------
    metrics dict with keys: rmse, mae, precision_at_k, recall_at_k,
                             coverage, diversity
    """
    from config import RECOMMEND_CFG as _DEFAULT_REC_CFG

    if rec_cfg is None:
        rec_cfg = _DEFAULT_REC_CFG

    out_dir = "reports"
    os.makedirs(out_dir, exist_ok=True)

    logger.info("=" * 50)
    logger.info("RECOMMENDER SYSTEM EVALUATION")
    logger.info("=" * 50)
    logger.info(
        "Popularity penalty: %.3f  |  Sample users: %d",
        rec_cfg.popularity_penalty, cfg.sample_users,
    )

    logger.info("Rating Accuracy  (%d val samples)", len(val_data))
    r = rmse(model, val_data)
    m = mae(model, val_data)
    logger.info("  RMSE : %.4f", r)
    logger.info("  MAE  : %.4f", m)

    logger.info("Ranking Metrics  (K=%d)", cfg.k)
    p_at_k, r_at_k = precision_recall_at_k(model, val_data, k=cfg.k, threshold=cfg.threshold)
    logger.info("  Precision@%d : %.4f", cfg.k, p_at_k)
    logger.info("  Recall@%d    : %.4f", cfg.k, r_at_k)

    logger.info("Diversity & Coverage  (%d sampled users)", cfg.sample_users)
    sampled_users   = list(user_map.keys())[: cfg.sample_users]
    idx_to_tmdb     = {v: k for k, v in movie_map.items()}
    rec_ids_list:   list[list[int]] = []
    all_recommended: set[int]       = set()

    # Build popularity penalty vector once — reused for all sampled users
    pop_penalty: np.ndarray | None = None
    if rec_cfg.popularity_penalty > 0.0 and ratings is not None:
        from src.recommend import _build_popularity_scores
        pop_penalty = _build_popularity_scores(ratings, movie_map) * rec_cfg.popularity_penalty
        logger.info(
            "  Applying popularity penalty (w=%.3f) to coverage scoring",
            rec_cfg.popularity_penalty,
        )

    for uid in tqdm(sampled_users, desc="  Scoring users", unit="user", leave=False):
        u_idx  = user_map[uid]
        scores = model.predict_user_all(u_idx)
        if pop_penalty is not None:
            scores = scores - pop_penalty
        top_k_idxs = np.argsort(scores)[::-1][: cfg.k]
        top_k_tmdb = [idx_to_tmdb[i] for i in top_k_idxs if i in idx_to_tmdb]
        rec_ids_list.append(top_k_tmdb)
        all_recommended.update(top_k_tmdb)

    coverage  = catalog_coverage(all_recommended, len(movie_map))
    diversity = intra_list_diversity(rec_ids_list, movies)
    logger.info("  Coverage  : %.4f  (%.1f%% of catalog)", coverage, coverage * 100)
    logger.info("  Diversity : %.4f", diversity)

    metrics: dict[str, float] = {
        "rmse": r, "mae": m,
        "precision_at_k": p_at_k, "recall_at_k": r_at_k,
        "coverage": coverage, "diversity": diversity,
    }

    with open(f"{out_dir}/metrics.txt", "w") as f:
        f.write("Recommender System Evaluation\n")
        f.write("=" * 50 + "\n")
        f.write(f"Val samples      : {len(val_data):,}\n")
        f.write(f"Sample users     : {cfg.sample_users}\n")
        f.write(f"Pop penalty      : {rec_cfg.popularity_penalty:.3f}\n")
        f.write(f"RMSE             : {r:.4f}\n")
        f.write(f"MAE              : {m:.4f}\n")
        f.write(f"Precision@{cfg.k}     : {p_at_k:.4f}\n")
        f.write(f"Recall@{cfg.k}        : {r_at_k:.4f}\n")
        f.write(f"Coverage         : {coverage:.4f}\n")
        f.write(f"Diversity        : {diversity:.4f}\n")

    logger.info("Generating charts...")
    _plot_convergence(model, out_dir)
    _plot_metrics_summary(metrics, cfg.k, out_dir)
    _plot_coverage_diversity(metrics, out_dir)
    _plot_rating_distribution(model, val_data, out_dir)

    logger.info("All charts + metrics saved to reports/")
    return metrics