SHELL := bash
VENV_NAME = .venv
# Find system Python, bypassing version managers (asdf, pyenv, etc.)
# Checks /usr/bin first (system Python), then falls back to PATH
PYTHON ?= $(shell test -x /usr/bin/python3.12 && echo /usr/bin/python3.12 || test -x /usr/bin/python3 && echo /usr/bin/python3 || command -v python3 2>/dev/null || echo python3)
ACTIVATE = . $(VENV_NAME)/bin/activate

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
	@echo "=== Setting up development environment ==="
	@if command -v asdf >/dev/null 2>&1 && ! test -x "$(PYTHON)"; then \
		echo "⚠️  asdf detected but no Python version set"; \
		echo ""; \
		echo "Option 1 - Use system Python (recommended):"; \
		echo "  export PATH=/usr/bin:\$$PATH"; \
		echo "  make setup"; \
		echo ""; \
		echo "Option 2 - Install Python 3.12 system-wide:"; \
		echo "  sudo apt install python3.12 python3.12-venv libsqlite3-dev"; \
		echo "  make setup PYTHON=/usr/bin/python3.12"; \
		echo ""; \
		exit 1; \
	fi
	@echo "Using Python: $(PYTHON)"
	@$(PYTHON) --version || (echo "Error: Python not found at $(PYTHON)" && exit 1)
	@$(PYTHON) -m venv $(VENV_NAME)
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
