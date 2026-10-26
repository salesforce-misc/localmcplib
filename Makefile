# localmcplib developer and CI tasks use the project's locked environment.
# `make` with no target prints help.

.DEFAULT_GOAL := help
.PHONY: help install format format-check lint lint-fix typecheck test test-seatbelt test-bubblewrap coverage build clean ci

# Require the host's real sandbox instead of allowing enforcement tests to skip.
HOST_OS := $(shell uname -s)
SANDBOX_TEST_ENV :=
ifeq ($(HOST_OS),Darwin)
SANDBOX_TEST_ENV := LOCALMCP_REQUIRE_SEATBELT=1
else ifeq ($(HOST_OS),Linux)
SANDBOX_TEST_ENV := LOCALMCP_REQUIRE_BUBBLEWRAP=1
endif

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Sync the venv from the lockfile (incl. dev deps)
	uv sync --locked

format: ## Format code with ruff
	uv run ruff format .

format-check: ## Check formatting without modifying files
	uv run ruff format --check .

lint: ## Lint with ruff
	uv run ruff check .

lint-fix: ## Lint and apply safe autofixes
	uv run ruff check --fix .

typecheck: ## Type-check the package with mypy
	uv run mypy src

test: ## Run the test suite, requiring the native OS sandbox
	$(SANDBOX_TEST_ENV) uv run pytest

test-seatbelt: ## Run real Seatbelt enforcement tests (macOS only; fails instead of skipping)
	LOCALMCP_REQUIRE_SEATBELT=1 uv run pytest -m seatbelt -v

test-bubblewrap: ## Run real bubblewrap enforcement tests (Linux only; fails instead of skipping)
	LOCALMCP_REQUIRE_BUBBLEWRAP=1 uv run pytest -m bubblewrap -v

coverage: ## Run tests with the native OS sandbox, coverage, and coverage.xml
	$(SANDBOX_TEST_ENV) uv run pytest --cov=localmcp --cov-report=term-missing --cov-report=xml:coverage.xml --cov-fail-under=90

build: ## Build the sdist and wheel into dist/
	uv build

clean: ## Remove build artifacts and caches
	rm -rf dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage coverage.xml htmlcov

ci: format-check lint typecheck coverage ## Run all checks and emit coverage
