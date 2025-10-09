.PHONY: help setup lint typecheck test ingest build-index query

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup:  ## Create venv & install dependencies
	uv venv && uv pip install -e .[dev]

lint:  ## Run ruff linter
	ruff check src/ scripts/

typecheck:  ## Run mypy type checker
	mypy src/

test:  ## Run pytest
	pytest

ingest:  ## Run sample ingest
	python scripts/ingest_sample.py

build-index:  ## Build FAISS index
	python scripts/build_index.py

query:  ## Run a sample query
	python scripts/query.py