# ──────────────────────────────────────────────────────────────────────────────
# Makefile — MovieMatcher
#
# Targets
# -------
#   make install        install all dependencies
#   make train          train MF model (full dataset)
#   make train-small    train MF model (dev mode — fast)
#   make train-bpr      train BPR model (full dataset)
#   make eval           run evaluation on latest trained model
#   make demo           launch Gradio demo locally
#   make test           run all unit tests with coverage report
#   make lint           run ruff + mypy type checks
#   make clean          remove generated artefacts (models, reports, caches)
#   make prep-demo-data generate slim parquet files for the Gradio app
# ──────────────────────────────────────────────────────────────────────────────

.PHONY: install train train-small train-bpr eval demo test lint clean prep-demo-data

PYTHON      ?= python
PIP         ?= pip
USER_ID     ?= 1
TOP_N       ?= 10

# ── Dependencies ──────────────────────────────────────────────────────────────
install:
	$(PIP) install -r requirements.txt
	@echo "✅ Dependencies installed"

# ── Training ──────────────────────────────────────────────────────────────────
train:
	$(PYTHON) main.py --user $(USER_ID) --topn $(TOP_N)

train-small:
	$(PYTHON) main.py --small --user $(USER_ID) --topn $(TOP_N)

train-bpr:
	$(PYTHON) main.py --model bpr --user $(USER_ID) --topn $(TOP_N)

train-bpr-small:
	$(PYTHON) main.py --model bpr --small --user $(USER_ID) --topn $(TOP_N)

train-hybrid:
	$(PYTHON) main.py --small --hybrid --user $(USER_ID) --topn $(TOP_N)

# ── Evaluation only (requires trained model + val_data) ───────────────────────
eval:
	@echo "Run 'make train' or 'make train-small' first to generate val_data."
	@echo "Evaluation runs automatically at the end of training."

# ── Gradio demo ───────────────────────────────────────────────────────────────
demo:
	$(PYTHON) app.py

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

test-fast:
	$(PYTHON) -m pytest tests/ -v --tb=short -x   # stop on first failure

# ── Linting & type-checking ───────────────────────────────────────────────────
lint:
	$(PYTHON) -m ruff check . --select=E,W,F,I
	$(PYTHON) -m mypy src/ --ignore-missing-imports --pretty

# ── Data prep for Gradio demo ─────────────────────────────────────────────────
prep-demo-data:
	$(PYTHON) scripts/prep_demo_data.py

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	rm -rf models/*.pkl
	rm -rf reports/*.png reports/*.txt reports/*.npy reports/*.log
	rm -rf __pycache__ src/__pycache__ tests/__pycache__
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf reports/cosine_sim_*.npy
	@echo "✅ Clean complete"

clean-cache:
	rm -rf reports/cosine_sim_*.npy
	@echo "✅ Similarity cache cleared"