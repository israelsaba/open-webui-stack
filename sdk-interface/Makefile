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

.PHONY: setup run generate-token clean nvim repl 
	
help:
	@echo "Makefile commands:"
	@echo "  setup          - use asdf, create venv, and install deps"
	@echo "  run  	        - starts the server" 
	@echo "  generate-token - generate a secure API token for authentication"
	@echo "  clean          - remove virtual environment and *.pyc/__pycache__"
	@echo "  nvim           - open Neovim in project with venv active"
	@echo "  repl           - start Python REPL with venv activated"
	@echo ""
	@echo "You can override ENV=dev to switch environments."

setup:
	@command -v asdf >/dev/null 2>&1 || { echo "asdf not found. Install asdf first: https://asdf-vm.com"; exit 1; }
	@PY_VER=$$(awk '/^python /{print $$2}' .tool-versions); \
	echo "Using asdf python $$PY_VER"; \
	asdf plugin-add python >/dev/null 2>&1 || true; \
	asdf install; \
	ASDF_PY=$$(asdf which python); \
	echo "Creating venv with $$ASDF_PY"; \
	$$ASDF_PY -m venv $(VENV_NAME)
ifeq ($(ENV),dev)
	$(ACTIVATE) && pip install --upgrade pip && pip install -r requirements-dev.txt
else
	$(ACTIVATE) && pip install --upgrade pip && pip install -r requirements.txt
endif

run:
	$(ACTIVATE) && uvicorn app.main:app --host 0.0.0.0 --port 8060 --reload --reload-exclude 'tests/*' --reload-exclude 'scripts/*'

clean:
	rm -rf $(VENV_NAME)
	- find . -type f -name "*.pyc" -delete 2>/dev/null || true
	- find . -type d -name "__pycache__" -delete 2>/dev/null || true

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
