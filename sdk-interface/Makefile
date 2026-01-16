SHELL := bash
ENV = dev
VENV_NAME = .venv
# Default to asdf-managed shim `python`. Override with `make PYTHON=python3.13` if needed.
PYTHON ?= python
ACTIVATE = . $(VENV_NAME)/bin/activate
TEST_PATH = tests

# ============================
# Windows users
# You can override PYTHON to python and adjust activate below accordingly
# ============================

.PHONY: help venv install run clean docker-up docker-down docker-build docker-run nvim test code repl setup generate-token

help:
	@echo "Makefile commands:"
	@echo "  setup          - use asdf, create venv, and install deps"
	@echo "  generate-token - generate a secure API token for authentication"
	@echo "  clean          - remove virtual environment and *.pyc/__pycache__"
	@echo "  docker-up      - start docker containers"
	@echo "  docker-down    - stop docker containers"
	@echo "  docker-build   - build development docker image"
	@echo "  docker-run     - open bash shell in backend container"
	@echo "  nvim           - open Neovim in project with venv active"
	@echo "  repl           - start Python REPL with venv activated"
	@echo ""
	@echo "You can override ENV=dev to switch environments."

install:
ifeq ($(ENV),dev)
	$(ACTIVATE) && pip install --upgrade pip && pip install -r requirements-dev.txt
else
	$(ACTIVATE) && pip install --upgrade pip && pip install -r requirements.txt
endif

setup:
	@command -v asdf >/dev/null 2>&1 || { echo "asdf not found. Install asdf first: https://asdf-vm.com"; exit 1; }
	@PY_VER=$$(awk '/^python /{print $$2}' .tool-versions); \
	echo "Using asdf python $$PY_VER"; \
	asdf plugin-add python >/dev/null 2>&1 || true; \
	asdf install python $$PY_VER; \
	asdf local python $$PY_VER; \
	ASDF_PY=$$(asdf which python); \
	echo "Creating venv with $$ASDF_PY"; \
	$$ASDF_PY -m venv $(VENV_NAME)
ifeq ($(ENV),dev)
	$(ACTIVATE) && pip install --upgrade pip && pip install -r requirements-dev.txt
else
	$(ACTIVATE) && pip install --upgrade pip && pip install -r requirements.txt
endif

run:
	$(ACTIVATE) && uvicorn app.main:app --host 0.0.0.0 --port 8060 --reload

clean:
	rm -rf $(VENV_NAME)
	- find . -type f -name "*.pyc" -delete 2>/dev/null || true
	- find . -type d -name "__pycache__" -delete 2>/dev/null || true


# Docker Commands
docker-up:
	docker compose up -d


docker-down:
	docker compose down


docker-build:
	docker compose build


test:
	@if [ -d "$(TEST_PATH)" ]; then \
		$(ACTIVATE) && python -m pytest $(TEST_PATH); \
	else \
		echo "No tests directory found. Skipping tests."; \
	fi

nvim:
	$(ACTIVATE) && nvim .

code:
	$(ACTIVATE) && code .

repl:
	$(ACTIVATE) && python -i

generate-token:
	$(ACTIVATE) && python scripts/generate_token.py
