.PHONY: help setup clean test test-cov test-real lint format run dev logs up down restart nvim vscode cursor

# ============================================================================
# Configuration
# ============================================================================

# Load environment from root .env if it exists
ifneq (,$(wildcard .env))
    include .env
    export
endif

# Detect if we're in test mode
TEST_MODE := $(or $(SDK__TEST_MODE),mock)

# ============================================================================
# Help
# ============================================================================

help: ## Show this help message
	@echo "Open WebUI Stack - Root Makefile"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Environment:"
	@echo "  Test Mode: $(TEST_MODE)"
	@echo "  .env file: $(if $(wildcard .env),✓ found,✗ not found)"
	@echo "  .env.test file: $(if $(wildcard .env.test),✓ found,✗ not found)"

# ============================================================================
# Setup
# ============================================================================

setup: ## Set up the development environment
	@echo "=== Setting up Open WebUI Stack ==="
	@if [ ! -f .env ]; then \
		echo "Creating .env from .env.example..."; \
		cp .env.example .env; \
		echo "⚠️  Please edit .env and add your API keys"; \
	else \
		echo "✓ .env already exists"; \
	fi
	@echo ""
	@echo "Setting up sdk-interface..."
	@$(MAKE) -C sdk-interface setup ENV=dev
	@echo ""
	@echo "✓ Setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Edit .env and add your API keys (SDK__GOOGLE_API_KEY, etc.)"
	@echo "  2. Run 'make test' to verify setup"
	@echo "  3. Run 'make up' to start services"

setup-test: ## Set up test environment
	@echo "=== Setting up Test Environment ==="
	@if [ ! -f .env.test ]; then \
		echo "Creating .env.test from .env.test.example..."; \
		cp .env.test.example .env.test; \
		echo "✓ Created .env.test (using mock mode by default)"; \
	else \
		echo "✓ .env.test already exists"; \
	fi
	@$(MAKE) -C sdk-interface setup ENV=test
	@echo "✓ Test environment ready"

clean: ## Clean all build artifacts and virtual environments
	@echo "=== Cleaning up ==="
	@$(MAKE) -C sdk-interface clean
	@echo "✓ Cleanup complete"

# ============================================================================
# Testing
# ============================================================================

test: ## Run tests in mock mode (default, fast)
	@echo "=== Running Tests ($(TEST_MODE) mode) ==="
	@if [ -f .env.test ]; then \
		set -a && . .env.test && set +a && $(MAKE) -C sdk-interface test; \
	else \
		SDK__TEST_MODE=mock $(MAKE) -C sdk-interface test; \
	fi

test-cov: ## Run tests with coverage report
	@echo "=== Running Tests with Coverage ($(TEST_MODE) mode) ==="
	@if [ -f .env.test ]; then \
		set -a && . .env.test && set +a && $(MAKE) -C sdk-interface test-cov; \
	else \
		SDK__TEST_MODE=mock $(MAKE) -C sdk-interface test-cov; \
	fi

test-real: ## Run tests against real APIs (requires API keys in .env.test)
	@echo "=== Running Tests Against Real APIs ==="
	@if [ ! -f .env.test ]; then \
		echo "Error: .env.test not found. Run 'make setup-test' first."; \
		exit 1; \
	fi
	@SDK__TEST_MODE=real $(MAKE) -C sdk-interface test-cov

# ============================================================================
# Development
# ============================================================================

run: ## Run sdk-interface in development mode
	@echo "=== Starting SDK Interface (development mode) ==="
	@if [ ! -f .env ]; then \
		echo "Error: .env not found. Run 'make setup' first."; \
		exit 1; \
	fi
	@set -a && . .env && set +a && $(MAKE) -C sdk-interface run

dev: run ## Alias for 'run'

lint: ## Run linter on sdk-interface
	@$(MAKE) -C sdk-interface lint

format: ## Format code in sdk-interface
	@$(MAKE) -C sdk-interface format

# ============================================================================
# Docker Compose
# ============================================================================

up: ## Start all services with docker-compose
	@echo "=== Starting Services ==="
	@if [ ! -f .env ]; then \
		echo "Error: .env not found. Run 'make setup' first."; \
		exit 1; \
	fi
	docker compose up -d
	@echo ""
	@echo "✓ Services started"
	@echo "  Open WebUI: http://localhost:8090"
	@echo "  SDK Interface: http://localhost:8060 (internal)"
	@echo ""
	@echo "Run 'make logs' to view logs"

down: ## Stop all services
	@echo "=== Stopping Services ==="
	docker compose down
	@echo "✓ Services stopped"

restart: ## Restart all services
	@echo "=== Restarting Services ==="
	docker compose restart
	@echo "✓ Services restarted"

logs: ## View logs from all services
	docker compose logs -f

logs-sdk: ## View logs from sdk-interface only
	docker compose logs -f sdk-interface

logs-webui: ## View logs from open-webui only
	docker compose logs -f open-webui

# ============================================================================
# Utilities
# ============================================================================

ps: ## Show running containers
	docker compose ps

shell-sdk: ## Open shell in sdk-interface container
	docker compose exec sdk-interface sh

generate-token: ## Generate a new API token for authentication
	@$(MAKE) -C sdk-interface generate-token

# ============================================================================
# Editors (with activated environment)
# ============================================================================

nvim: ## Open nvim with activated Python environment
	@echo "=== Opening nvim with activated environment ==="
	@if [ ! -d sdk-interface/.venv ]; then \
		echo "Error: Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi
	@cd sdk-interface && \
		bash -c 'source .venv/bin/activate && exec nvim'

vscode: ## Open VS Code with activated Python environment
	@echo "=== Opening VS Code ==="
	@if [ ! -d sdk-interface/.venv ]; then \
		echo "Error: Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi
	@code .

cursor: ## Open Cursor with activated Python environment
	@echo "=== Opening Cursor ==="
	@if [ ! -d sdk-interface/.venv ]; then \
		echo "Error: Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi
	@cursor .
