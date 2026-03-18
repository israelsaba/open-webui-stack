#!/bin/bash
# Virtual environment setup script for sdk-interface

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

echo "=== SDK Interface Virtual Environment Setup ==="
echo "Project directory: $PROJECT_DIR"
echo

# Check Python version
PYTHON_CMD="python3"
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version | cut -d' ' -f2)
echo "Using Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    echo "✓ Virtual environment created"
else
    echo "Virtual environment already exists at $VENV_DIR"
fi

# Activate virtual environment
echo
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies based on argument
INSTALL_TYPE="${1:-dev}"

case "$INSTALL_TYPE" in
    prod|production)
        echo
        echo "Installing production dependencies..."
        pip install -r "$PROJECT_DIR/requirements.txt"
        ;;
    dev|development)
        echo
        echo "Installing development dependencies..."
        pip install -r "$PROJECT_DIR/requirements-dev.txt"
        ;;
    test|testing)
        echo
        echo "Installing testing dependencies..."
        pip install -r "$PROJECT_DIR/requirements-test.txt"
        ;;
    all)
        echo
        echo "Installing all dependencies..."
        pip install -r "$PROJECT_DIR/requirements.txt"
        pip install -r "$PROJECT_DIR/requirements-dev.txt"
        pip install -r "$PROJECT_DIR/requirements-test.txt"
        ;;
    *)
        echo "Error: Invalid installation type '$INSTALL_TYPE'"
        echo "Usage: $0 [prod|dev|test|all]"
        exit 1
        ;;
esac

echo
echo "✓ Dependencies installed successfully"
echo
echo "=== Setup Complete ==="
echo
echo "To activate the virtual environment, run:"
echo "  source $VENV_DIR/bin/activate"
echo
echo "To deactivate, run:"
echo "  deactivate"
echo
