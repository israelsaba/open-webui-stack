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

.PHONY: setup run generate-token clean nvim repl test lint lock
	
help:
	@echo "Makefile commands:"
	@echo "  setup          - prepare environment for run or test"
	@echo "  run            - starts the server locally with hot reload" 
	@echo "  test           - run tests (same as CI/CD)"
	@echo "  lint           - fix all lint issues (safe + unsafe) and test with coverage"
	@echo "  lock           - regenerate requirements lock files"
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
	@echo "Installing all dependencies from lock files..."
	$(ACTIVATE) && pip install --upgrade pip pip-tools
	$(ACTIVATE) && pip install -r requirements.lock
	$(ACTIVATE) && pip install -r requirements-dev.txt
	$(ACTIVATE) && pip install -r requirements-test.lock
	@echo "✓ Setup complete! Activate with: source $(VENV_NAME)/bin/activate"

lock:
	@echo "=== Regenerating lock files ==="
	@if [ ! -d "$(VENV_NAME)" ]; then \
		echo "Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi
	$(ACTIVATE) && pip install --upgrade pip-tools
	$(ACTIVATE) && pip-compile requirements.txt -o requirements.lock --strip-extras
	$(ACTIVATE) && pip-compile requirements-test.txt -o requirements-test.lock --strip-extras
	@echo "✓ Lock files regenerated!"

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
		$(ACTIVATE) && python -m pytest tests/ -v; \
	else \
		echo "No tests directory found. Skipping tests."; \
	fi

lint:
	@echo "=== Formatting code ==="
	$(ACTIVATE) && ruff format app tests
	@echo "=== Fixing lint issues (safe + unsafe) ==="
	$(ACTIVATE) && ruff check --fix --unsafe-fixes app tests
	@echo "=== Running tests with coverage ==="
	$(ACTIVATE) && python -m pytest tests/ -v -m "not slow" --cov=app --cov-report=term-missing --cov-report=html --cov-report=xml
	@echo "✓ All checks passed!"

nvim:
	$(ACTIVATE) && nvim .

code:
	$(ACTIVATE) && code .

repl:
	$(ACTIVATE) && python -i

generate-token:
	$(ACTIVATE) && python scripts/generate_token.py
