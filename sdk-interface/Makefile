SHELL := bash
ENV = dev
VENV_NAME = .venv
# Default to asdf-managed shim `python`. Override with `make PYTHON=python3.13` if needed.
PYTHON ?= python
ACTIVATE = . $(VENV_NAME)/bin/activate

# ============================
# Windows users
# You can override PYTHON to python and adjust activate below accordingly
# ============================

.PHONY: setup setup-dev setup-test run generate-token clean nvim repl test test-cov test-watch lint format 
	
help:
	@echo "Makefile commands:"
	@echo "  setup          - create venv and install ALL dependencies (dev + test)"
	@echo "  setup-dev      - create venv and install dev dependencies"
	@echo "  setup-test     - create venv and install test dependencies"
	@echo "  run            - starts the server locally with hot reload" 
	@echo "  test           - run tests (same as CI/CD)"
	@echo "  test-cov       - run tests with coverage report"
	@echo "  test-watch     - run tests in watch mode"
	@echo "  lint           - run linter (ruff)"
	@echo "  format         - format code with ruff"
	@echo "  generate-token - generate a secure API token for authentication"
	@echo "  clean          - remove virtual environment and *.pyc/__pycache__"
	@echo "  nvim           - open Neovim in project with venv active"
	@echo "  repl           - start Python REPL with venv activated"

setup:
	@echo "=== Setting up development environment with ALL dependencies ==="
	@if command -v asdf >/dev/null 2>&1; then \
		PY_VER=$$(awk '/^python /{print $$2}' .tool-versions); \
		echo "Using asdf python $$PY_VER"; \
		asdf plugin-add python >/dev/null 2>&1 || true; \
		asdf install; \
		ASDF_PY=$$(asdf which python); \
		echo "Creating venv with $$ASDF_PY"; \
		$$ASDF_PY -m venv $(VENV_NAME); \
	else \
		echo "asdf not found, using system python"; \
		$(PYTHON) -m venv $(VENV_NAME); \
	fi
	@echo "Installing all dependencies..."
	$(ACTIVATE) && pip install --upgrade pip
	$(ACTIVATE) && pip install -r requirements.txt
	$(ACTIVATE) && pip install -r requirements-dev.txt
	$(ACTIVATE) && pip install -r requirements-test.txt
	@echo "✓ Setup complete! Activate with: source $(VENV_NAME)/bin/activate"

setup-dev:
	@echo "=== Setting up development environment ==="
	@if command -v asdf >/dev/null 2>&1; then \
		PY_VER=$$(awk '/^python /{print $$2}' .tool-versions); \
		echo "Using asdf python $$PY_VER"; \
		asdf plugin-add python >/dev/null 2>&1 || true; \
		asdf install; \
		ASDF_PY=$$(asdf which python); \
		echo "Creating venv with $$ASDF_PY"; \
		$$ASDF_PY -m venv $(VENV_NAME); \
	else \
		echo "asdf not found, using system python"; \
		$(PYTHON) -m venv $(VENV_NAME); \
	fi
	$(ACTIVATE) && pip install --upgrade pip && pip install -r requirements-dev.txt
	@echo "✓ Development setup complete!"

setup-test:
	@echo "=== Setting up test environment ==="
	@if command -v asdf >/dev/null 2>&1; then \
		PY_VER=$$(awk '/^python /{print $$2}' .tool-versions); \
		echo "Using asdf python $$PY_VER"; \
		asdf plugin-add python >/dev/null 2>&1 || true; \
		asdf install; \
		ASDF_PY=$$(asdf which python); \
		echo "Creating venv with $$ASDF_PY"; \
		$$ASDF_PY -m venv $(VENV_NAME); \
	else \
		echo "asdf not found, using system python"; \
		$(PYTHON) -m venv $(VENV_NAME); \
	fi
	$(ACTIVATE) && pip install --upgrade pip && pip install -r requirements-test.txt
	@echo "✓ Test setup complete!"

run:
	$(ACTIVATE) && uvicorn app.main:app --host 0.0.0.0 --port 8060 --reload --reload-exclude 'tests/*' --reload-exclude 'scripts/*'

clean:
	@echo "=== Cleaning up ==="
	rm -rf $(VENV_NAME)
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf coverage.xml
	rm -rf .ruff_cache
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	find . -type f -name '*.pyo' -delete 2>/dev/null || true
	find . -type f -name '*~' -delete 2>/dev/null || true
	@echo "✓ Cleanup complete!"

test:
	@if [ -d "tests" ]; then \
		if [ -f ".env.test" ]; then \
			echo "Loading .env.test for testing..."; \
			set -a && . .env.test && set +a && $(ACTIVATE) && python -m pytest tests/ -v; \
		else \
			echo "No .env.test found, using environment variables..."; \
			$(ACTIVATE) && python -m pytest tests/ -v; \
		fi \
	else \
		echo "No tests directory found. Skipping tests."; \
	fi

test-cov:
	@if [ -d "tests" ]; then \
		if [ -f ".env.test" ]; then \
			echo "Loading .env.test for testing..."; \
			set -a && . .env.test && set +a && $(ACTIVATE) && python -m pytest tests/ -v --cov=app --cov-report=term-missing --cov-report=html --cov-report=xml; \
		else \
			echo "No .env.test found, using environment variables..."; \
			$(ACTIVATE) && python -m pytest tests/ -v --cov=app --cov-report=term-missing --cov-report=html --cov-report=xml; \
		fi \
	else \
		echo "No tests directory found. Skipping tests."; \
	fi

test-watch:
	@if [ -d "tests" ]; then \
		$(ACTIVATE) && python -m pytest tests/ -v --watch; \
	else \
		echo "No tests directory found. Skipping tests."; \
	fi

lint:
	$(ACTIVATE) && ruff check app tests

format:
	$(ACTIVATE) && ruff format app tests
	$(ACTIVATE) && ruff check --fix app tests

nvim:
	$(ACTIVATE) && nvim .

code:
	$(ACTIVATE) && code .

repl:
	$(ACTIVATE) && python -i

generate-token:
	$(ACTIVATE) && python scripts/generate_token.py
