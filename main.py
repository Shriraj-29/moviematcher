"""
main.py — MovieMatcher pipeline entry-point.

Usage:
    python main.py                          # full ratings.csv, CF for user 1
    python main.py --small                  # dev mode (ratings_small.csv, fast)
    python main.py --user 42               # recommend for a specific user
    python main.py --nrows 500000          # cap rows loaded from full ratings.csv
    python main.py --small --hybrid        # hybrid CF+CB recommendations
    python main.py --small --model bpr     # use BPR ranking model instead of MF
    python main.py --small --verbose       # debug-level logging

Logs are written to stdout (INFO by default) and to reports/run.log.
"""

import argparse
import logging
import os
import sys

from config import MF_CFG, BPR_CFG, DATA_CFG, EVAL_CFG, RECOMMEND_CFG, REPORTS_DIR
from src.data_loader import load_data
from src.train import train_model, train_bpr_model, save_model
from src.recommend import get_top_n
from src.evaluate import run_evaluation
from src.content_based import hybrid_recommend

import pickle
from pathlib import Path

# ── Logging setup ────────────────────────────────────────────────────────────

def setup_logging(verbose: bool = False) -> None:
    """Configure root logger: stdout + rotating file in reports/."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    fmt   = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(REPORTS_DIR / "run.log", mode="a", encoding="utf-8"),
    ]
    logging.basicConfig(level=level, format=fmt, handlers=handlers)


logger = logging.getLogger(__name__)


# ── Entry-point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MovieMatcher recommendation pipeline")
    parser.add_argument("--user",    type=int,  default=1,     help="User ID to recommend for")
    parser.add_argument("--topn",    type=int,  default=10,    help="Number of recommendations")
    parser.add_argument("--nrows",   type=int,  default=None,  help="Cap rows loaded (None=all)")
    parser.add_argument("--small",   action="store_true",      help="Use ratings_small.csv (dev mode)")
    parser.add_argument("--hybrid",  action="store_true",      help="Hybrid CF+CB recommendations")
    parser.add_argument("--model",   choices=["mf", "bpr"],    default="mf",
                        help="Model type: mf (Matrix Factorisation) or bpr (Bayesian PR)")
    parser.add_argument("--retrain", action="store_true",      help="Force retrain even if a saved model exists")
    parser.add_argument("--verbose", action="store_true",      help="Debug-level logging")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    logger.info("MovieMatcher starting | user=%d model=%s hybrid=%s small=%s",
                args.user, args.model, args.hybrid, args.small)

    # ── Load ─────────────────────────────────────────────────────────────────
    ratings, movies = load_data(data_dir="data", small=args.small, nrows=args.nrows)
    logger.info(
        "Dataset: %d ratings | %d movies | %d users",
        len(ratings), len(movies), ratings["userId"].nunique(),
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    model_path = Path("models/bpr.pkl" if args.model == "bpr" else "models/mf.pkl")
    if model_path.exists() and not args.retrain:
        logger.info("Loading existing %s model from %s...", args.model.upper(), model_path)
        with open(model_path, "rb") as f:
            saved = pickle.load(f)
        # MF saves (model, user_map, movie_map, val_data); BPR saves (model, user_map, movie_map)
        if len(saved) == 4:
            model, user_map, movie_map, val_data = saved
        else:
            model, user_map, movie_map = saved
            val_data = []
    else:
        if args.model == "bpr":
            logger.info("Training BPR model (k=%d lr=%g reg=%g epochs=%d)...",
                        BPR_CFG.k, BPR_CFG.lr, BPR_CFG.reg, BPR_CFG.epochs)
            model, user_map, movie_map = train_bpr_model(ratings, cfg=BPR_CFG, data_cfg=DATA_CFG)
            save_model((model, user_map, movie_map), path=str(model_path))
            val_data = []
        else:
            logger.info("Training MF model (k=%d lr=%g reg=%g epochs=%d)...",
                        MF_CFG.k, MF_CFG.lr, MF_CFG.reg, MF_CFG.epochs)
            model, user_map, movie_map, val_data = train_model(ratings, cfg=MF_CFG, data_cfg=DATA_CFG)
            save_model((model, user_map, movie_map, val_data), path=str(model_path))

    # ── Evaluate ──────────────────────────────────────────────────────────────
    if val_data:
        logger.info("Running evaluation...")
        run_evaluation(model, val_data, user_map, movie_map, movies, ratings=ratings, cfg=EVAL_CFG, rec_cfg=RECOMMEND_CFG)
    else:
        logger.info("Skipping rating-based evaluation for BPR (ranking model).")

    # ── Recommend ─────────────────────────────────────────────────────────────
    mode = "Hybrid (CF + CB)" if args.hybrid else (
        "Bayesian PR" if args.model == "bpr" else "Collaborative Filtering"
    )
    logger.info("Top-%d %s recommendations for User %d:", args.topn, mode, args.user)
    print(f"\nTop-{args.topn} {mode} recommendations for User {args.user}:")
    print("-" * 50)

    try:
        if args.hybrid:
            recs = hybrid_recommend(
                model, args.user, movies, ratings, user_map, movie_map, n=args.topn
            )
        else:
            recs = get_top_n(
                model, args.user, movies, ratings, user_map, movie_map, n=args.topn
            )
        for rank, (title, score) in enumerate(recs, 1):
            print(f"  {rank:2}. {title}  ({score:.3f})")
    except ValueError as e:
        logger.error("Recommendation failed: %s", e)
        print(f"  Error: {e}")


if __name__ == "__main__":
    main()