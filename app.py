"""
app.py — MovieMatcher Gradio demo (HuggingFace Spaces entry-point).

Architecture
------------
All assets are loaded SYNCHRONOUSLY at startup before Gradio binds to a port.
No background threads, no lazy-loading flags, no polling timer.

The app either starts ready in <5 s (parquet + pickle + pre-built .npz load)
or exits with a clear error. Tab switching is pure client-side JS — zero server
round-trips — so it can never freeze regardless of server load.

Offline prep (run once locally, commit artefacts):
    python scripts/prep_demo_data.py   # writes movies_slim, ratings_slim, tfidf.npz
    python main.py --small             # trains and saves models/mf.pkl
    huggingface-cli upload <repo> models/mf.pkl models/mf.pkl
    huggingface-cli upload <repo> data/tfidf.npz data/tfidf.npz
"""

from __future__ import annotations

# ── Force non-interactive matplotlib backend BEFORE any other import ──────────
import matplotlib
matplotlib.use("Agg")

import logging
import os
import pickle
import time
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import scipy.sparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _t(label: str, t0: float) -> None:
    """Log a step label and ms elapsed since t0."""
    logger.debug("  %-50s  %.0f ms", label, (time.perf_counter() - t0) * 1000)


# ── Paths (override via env vars for HF Spaces) ───────────────────────────────

MODEL_PATH   = Path(os.getenv("MODEL_PATH",   "models/mf.pkl"))
MOVIES_PATH  = Path(os.getenv("MOVIES_PATH",  "data/movies_slim.parquet"))
RATINGS_PATH = Path(os.getenv("RATINGS_PATH", "data/ratings_slim.parquet"))
TFIDF_PATH   = Path(os.getenv("TFIDF_PATH",   "data/tfidf.npz"))


# ── Synchronous startup load — runs before Gradio binds to any port ───────────

def _load_all():
    """
    Load every asset synchronously.  Raises on any failure so the process
    exits immediately with a clear traceback instead of serving a broken UI.
    """
    t_total = time.perf_counter()

    # ── HF Hub download if needed ────────────────────────────────────────────
    if not MODEL_PATH.exists():
        repo = os.getenv("HF_REPO")
        if repo:
            t = time.perf_counter()
            logger.info("[startup] Downloading model from HF Hub (%s)...", repo)
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo_id=repo, filename="mf.pkl", local_dir="models")
            _t("[startup] HF Hub download", t)
        else:
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run `python main.py --small` to train, or set HF_REPO."
            )

    # ── Model pickle ─────────────────────────────────────────────────────────
    t = time.perf_counter()
    logger.info("[startup] Loading model from %s ...", MODEL_PATH)
    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    model, user_map, movie_map = saved[:3]
    _t("[startup] pickle.load(model)", t)
    logger.debug("  users=%d  items=%d", len(user_map), len(movie_map))

    # ── Movies parquet ────────────────────────────────────────────────────────
    t = time.perf_counter()
    logger.info("[startup] Loading movies from %s ...", MOVIES_PATH)
    movies = pd.read_parquet(MOVIES_PATH)
    _t("[startup] read_parquet(movies)", t)
    logger.debug("  rows=%d  cols=%s", len(movies), list(movies.columns))

    # ── Ratings parquet ───────────────────────────────────────────────────────
    t = time.perf_counter()
    logger.info("[startup] Loading ratings from %s ...", RATINGS_PATH)
    ratings = pd.read_parquet(RATINGS_PATH)
    _t("[startup] read_parquet(ratings)", t)
    logger.debug("  rows=%d", len(ratings))

    # ── TF-IDF sparse matrix ──────────────────────────────────────────────────
    if not TFIDF_PATH.exists():
        raise FileNotFoundError(
            f"TF-IDF matrix not found at {TFIDF_PATH}. "
            "Run `python scripts/prep_demo_data.py` to generate it."
        )
    t = time.perf_counter()
    logger.info("[startup] Loading TF-IDF matrix from %s ...", TFIDF_PATH)
    tfidf_matrix = scipy.sparse.load_npz(str(TFIDF_PATH)).astype(np.float32)
    _t("[startup] load_npz(tfidf)", t)
    logger.debug("  shape=%s  nnz=%d  dtype=%s",
                 tfidf_matrix.shape, tfidf_matrix.nnz, tfidf_matrix.dtype)

    _t("[startup] TOTAL _load_all()", t_total)
    logger.info(
        "[startup] All assets ready: %d movies | %d users | %d items | tfidf %s",
        len(movies), len(user_map), len(movie_map), tfidf_matrix.shape,
    )
    return model, user_map, movie_map, movies, ratings, tfidf_matrix

logger.info("[startup] Gradio version: %s", gr.__version__)
t0 = time.perf_counter()
logger.info("[startup] Loading assets (blocking until complete)...")
_model, _user_map, _movie_map, _movies, _ratings, _tfidf_matrix = _load_all()
logger.info("[startup] Done in %.1f s — launching Gradio.", time.perf_counter() - t0)

# ── Pre-warm every lazy import that could stall the first user request ────────
logger.info("[startup] Pre-warming src imports...")
_t0_warm = time.perf_counter()
import matplotlib
from src.recommend import get_top_n, _build_popularity_scores
from src.content_based import get_similar_movies, hybrid_recommend
_t("[startup] import pre-warm complete", _t0_warm)


# ── Tab 1 — CF / Hybrid recommendations ──────────────────────────────────────

def recommend_for_user(user_id: int, mode: str, top_n: int) -> str:
    t_total = time.perf_counter()
    logger.debug("[recommend] called  user_id=%s  mode=%s  top_n=%s", user_id, mode, top_n)

    try:
        user_id = int(user_id)
    except (ValueError, TypeError):
        return "Please enter a valid integer User ID."

    if mode == "Hybrid (CF + CB)":
        t = time.perf_counter()
        logger.debug("[recommend] entering hybrid_recommend...")
        try:
            recs = hybrid_recommend(
                _model, user_id, _movies, _ratings,
                _user_map, _movie_map,
                n=int(top_n),
                tfidf_matrix=_tfidf_matrix,
            )
        except ValueError as e:
            return f"Error: {e}"
        _t("[recommend] hybrid_recommend()", t)
    else:
        t = time.perf_counter()
        logger.debug("[recommend] entering get_top_n...")
        try:
            recs = get_top_n(
                _model, user_id, _movies, _ratings,
                _user_map, _movie_map, n=int(top_n),
            )
        except ValueError as e:
            return f"Error: {e}"
        _t("[recommend] get_top_n()", t)

    _t("[recommend] TOTAL recommend_for_user()", t_total)

    if not recs:
        return "No recommendations found for this user."

    lines = [f"🎬 Top-{top_n} {mode} recommendations for User {user_id}\n"]
    for rank, (title, score) in enumerate(recs, 1):
        lines.append(f"  {rank:2}. {title:<45}  score: {score:.3f}")
    return "\n".join(lines)


# ── Tab 2 — Similar movies ────────────────────────────────────────────────────

def find_similar(title: str, top_n: int) -> str:
    t_total = time.perf_counter()
    logger.debug("[similar] called  title=%r  top_n=%s", title, top_n)

    title = title.strip()
    if not title:
        return "Please enter a movie title."

    t = time.perf_counter()
    logger.debug("[similar] entering get_similar_movies...")
    try:
        recs = get_similar_movies(title, _movies, tfidf_matrix=_tfidf_matrix, n=int(top_n))
    except ValueError as e:
        return f"Error: {e}"
    _t("[similar] get_similar_movies()", t)
    _t("[similar] TOTAL find_similar()", t_total)

    lines = [f"🎥 Movies similar to '{title}'\n"]
    for rank, (t_title, score) in enumerate(recs, 1):
        lines.append(f"  {rank:2}. {t_title:<45}  similarity: {score:.4f}")
    return "\n".join(lines)


# ── Tab 3 — About ─────────────────────────────────────────────────────────────

ABOUT_TEXT = """
# 🎬 MovieMatcher

**A movie recommendation system built on the [Kaggle Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)**
(26 million ratings · 270,000 users · 45,000 movies)

---

## Models

| Model | Approach | Best for |
|-------|----------|----------|
| **Collaborative Filtering** | Matrix Factorisation (SGD + biases) | Warm users with rating history |
| **Hybrid (CF + CB)** | 0.7 × CF + 0.3 × TF-IDF cosine similarity | Balancing personalisation + diversity |
| **BPR** | Bayesian Personalised Ranking (pairwise) | Ranking quality over rating accuracy |

## Architecture

```
Prediction  = μ + b_u + b_i + P[u] · Q[i]
CF scoring  = P[u] @ Q.T  (one vectorised call — no Python loops)
CB scoring  = mean cosine-sim(item, top-5 rated by user)
Hybrid      = 0.7 × CF + 0.3 × CB
Pop penalty = score − 0.05 × log(count+1) / log(max_count+1)
```

## Evaluation Results (full dataset — 26M ratings)

| Metric | Score |
|--------|-------|
| RMSE | 0.7836 |
| MAE | 0.5941 |
| Precision@10 | 0.4407 |
| Recall@10 | 0.8768 |
| Coverage | 2.7% |
| Diversity (ILD) | 0.7187 |

## Tech Stack

Python · Pandas · NumPy · scikit-learn · Gradio · Matplotlib · tqdm

[GitHub Repository](https://github.com/Shriraj-29/moviematcher)
"""


# ── Build Gradio UI ───────────────────────────────────────────────────────────

def build_demo() -> gr.Blocks:
    with gr.Blocks(title="MovieMatcher") as demo:

        gr.Markdown("# 🎬 MovieMatcher\n*Collaborative Filtering · Content-Based · Hybrid*")

        with gr.Tabs():

            # ── Tab 1 — shorter name fits on mobile tab bar ────────────────────
            with gr.Tab("Recommend"):
                with gr.Column(elem_classes="tab-content"):
                    gr.Markdown("Enter a **User ID** (1–270,000) to get personalised recommendations.")
                    with gr.Row(elem_classes="input-row top-row"):
                        uid_input  = gr.Number(label="User ID", value=1, precision=0, scale=1, min_width=0)
                        topn_input = gr.Slider(5, 20, value=10, step=1, label="Top N", scale=1, min_width=0)
                    mode_input = gr.Radio(
                        ["Collaborative Filtering", "Hybrid (CF + CB)"],
                        value="Collaborative Filtering",
                        label="Model",
                    )
                    rec_btn    = gr.Button("Get Recommendations", variant="primary")
                    rec_output = gr.Textbox(label="Recommendations", lines=14, max_lines=20)
                    rec_btn.click(
                        recommend_for_user,
                        inputs=[uid_input, mode_input, topn_input],
                        outputs=rec_output,
                    )

            # ── Tab 2 ─────────────────────────────────────────────────────────
            with gr.Tab("Similar"):
                with gr.Column(elem_classes="tab-content"):
                    gr.Markdown("Find movies with similar genre / director / cast using **TF-IDF cosine similarity**.")
                    with gr.Row(elem_classes="input-row"):
                        title_input = gr.Textbox(
                            label="Movie Title",
                            placeholder="e.g. The Dark Knight",
                            scale=3,
                            min_width=0,
                        )
                        topn2_input = gr.Slider(5, 20, value=10, step=1, label="Top N", scale=1, min_width=0)
                    sim_btn    = gr.Button("Find Similar", variant="primary")
                    sim_output = gr.Textbox(label="Similar Movies", lines=14, max_lines=20)
                    sim_btn.click(
                        find_similar,
                        inputs=[title_input, topn2_input],
                        outputs=sim_output,
                    )

            # ── Tab 3 ─────────────────────────────────────────────────────────
            with gr.Tab("About"):
                with gr.Column(elem_classes="tab-content"):
                    gr.Markdown(ABOUT_TEXT)

    return demo


# ── Launch ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = build_demo()
    demo.queue(default_concurrency_limit=4)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(primary_hue="indigo"),
        css="""
            /* ── No horizontal scroll ────────────────────────────────────────*/
            html, body { overflow-x: hidden !important; }
            .gradio-container {
                max-width: 900px !important;
                width: 100% !important;
                margin: auto !important;
                padding: 0 16px !important;
                box-sizing: border-box !important;
                overflow-x: hidden !important;
            }

            /* ── Tab nav: scrollable so all tabs visible on any screen width ──
               overflow-x:auto lets the tab bar scroll horizontally on mobile
               instead of collapsing tabs into a "..." overflow menu.          */
            .tab-nav {
                overflow-x: auto !important;
                -webkit-overflow-scrolling: touch !important;
                scrollbar-width: none !important;      /* Firefox */
                white-space: nowrap !important;
                flex-wrap: nowrap !important;
            }
            .tab-nav::-webkit-scrollbar { display: none; } /* Chrome/Safari */

            /* ── Active tab visibility ────────────────────────────────────────*/
            .tab-nav button.selected {
                color: #ffffff !important;
                border-bottom: 3px solid #818cf8 !important;
                font-weight: 700 !important;
                opacity: 1 !important;
            }
            .tab-nav button:not(.selected) {
                color: rgba(255,255,255,0.55) !important;
                border-bottom: 3px solid transparent !important;
            }
            .tab-nav button:not(.selected):hover {
                color: rgba(255,255,255,0.85) !important;
            }

            /* ── Tab content ──────────────────────────────────────────────────*/
            .tab-content {
                width: 100% !important;
                box-sizing: border-box !important;
                overflow: hidden !important;
            }

            /* ── Input rows: side-by-side on desktop ──────────────────────────*/
            .tab-content .input-row {
                width: 100% !important;
                display: flex !important;
                flex-wrap: nowrap !important;
                gap: 8px !important;
                box-sizing: border-box !important;
                overflow: hidden !important;
            }
            .tab-content .input-row > * {
                min-width: 0 !important;
                flex-shrink: 1 !important;
                box-sizing: border-box !important;
            }

            /* ── Mobile (≤ 600px) ─────────────────────────────────────────────
               .top-row (User ID + Top N) stays as a 2-column row — both
               inputs are simple enough to sit side by side even on a phone.
               The Radio (Model) is already full-width below the row.
               Tab 2's title + slider also stay side-by-side (2 items fit).  */
            @media (max-width: 600px) {
                .gradio-container { padding: 0 10px !important; }

                /* Similar Movies: stack title above slider on very small screens */
                .tab-content .input-row:not(.top-row) {
                    flex-wrap: wrap !important;
                }
                .tab-content .input-row:not(.top-row) > * {
                    flex: 1 1 100% !important;
                    min-width: 0 !important;
                }
            }
        """,
    )