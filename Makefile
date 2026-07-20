.PHONY: install install-all install-spatial install-cog install-stac install-models install-local \
       dev setup setup-all test test-fast test-cov lint format pre-commit typecheck nox check build clean \
       serve-docs build-docs dbx-build dbx-upload help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Installation ──────────────────────────────────────────────────────────────

install: ## Install core dependencies only
	uv sync

install-all: ## Install all optional extras (spatial, cog, stac, models, local)
	uv pip install -e ".[all]"

install-spatial: ## Install spatial extras (rasterio, geopandas, shapely, osbng, pyproj)
	uv pip install -e ".[spatial]"

install-cog: ## Install COG conversion extras (rio-cogeo, rasterio)
	uv pip install -e ".[cog]"

install-stac: ## Install STAC extras (pystac, stac-geoparquet, defusedxml)
	uv pip install -e ".[stac]"

install-models: ## Install model extras (torch, torchvision, torchgeo) — local dev only
	uv pip install -e ".[models]"

install-local: ## Install local pipeline extras (rioxarray, xarray, dask)
	uv pip install -e ".[local]"

dev: ## Install dev + docs groups (linting, testing, docs, nox, pyrefly)
	uv sync --group dev --group docs

setup: dev ## Full dev setup: install deps + configure pre-commit hooks
	uv run pre-commit install


setup-all: install-all ## Full dev setup: install deps + configure pre-commit hooks
	uv run pre-commit install

# ── Code Quality ──────────────────────────────────────────────────────────────

pre-commit: ## Run all pre-commit hooks on all files
	uv run pre-commit run --all-files

lint: ## Run ruff linter only
	uv run ruff check src/ tests/

format: ## Run ruff formatter
	uv run ruff format src/ tests/

typecheck: ## Run pyrefly type checker
	uv run pyrefly check src/geoembed

check: pre-commit test ## Run pre-commit hooks (lint + format + typecheck) + tests

# ── Testing ───────────────────────────────────────────────────────────────────

test: ## Run test suite (excludes slow/gpu)
	uv run pytest tests/ -v -m "not slow and not gpu"

test-all: ## Run full test suite including slow tests
	uv run pytest tests/ -v

test-cov: ## Run tests with coverage report
	uv run pytest tests/ --cov=src/geoembed --cov-report=term-missing -m "not slow and not gpu"

# ── Nox ───────────────────────────────────────────────────────────────────────

nox: ## Run all default nox sessions (ruff, pyrefly, tests)
	uv run nox

nox-lint: ## Run nox lint session only
	uv run nox -s ruff

nox-typecheck: ## Run nox type-check session only
	uv run nox -s pyrefly

nox-test: ## Run nox test session (all Python versions)
	uv run nox -s tests

nox-coverage: ## Run nox coverage session
	uv run nox -s coverage

# ── Build ─────────────────────────────────────────────────────────────────────

build: ## Build wheel to dist/
	uv build --wheel --out-dir dist/

clean: ## Remove build artifacts and caches
	rm -rf dist/ site/ .pytest_cache/ .ruff_cache/ .mypy_cache/ .nox/ htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# ── Documentation ─────────────────────────────────────────────────────────────

serve-docs: ## Serve docs locally at localhost:8001
	uv run --group docs mkdocs serve -a localhost:8001

build-docs: ## Build static docs site to site/
	uv run --group docs mkdocs build

# ── Databricks ────────────────────────────────────────────────────────────────

dbx-build: ## Build wheel to deps/whl/ for Databricks deployment
	uv build --wheel --out-dir dist/

dbx-upload: ## Show command to upload wheel to UC Volume
	@echo "Upload: deps/whl/ → /Volumes/.../dependencies/whl/"
	@echo "Run: databricks fs cp deps/whl/*.whl dbfs:/Volumes/<catalog>/dependencies/whl/ --overwrite"
