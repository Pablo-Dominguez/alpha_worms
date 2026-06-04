#!/bin/bash
set -e # Exit immediately if any command fails

echo "=== [1/4] Installing Poetry via Official Installer ==="
curl -sSL https://install.python-poetry.org | python3 -

echo "=== [2/4] Configuring Environment PATH variables ==="
POETRY_BIN="/root/.local/bin"
# Update path for the current running script session
export PATH="$POETRY_BIN:$PATH"

# Persist the path changes so future SSH sessions see the 'poetry' command
if ! grep -q "$POETRY_BIN" ~/.bashrc; then
    echo "export PATH=\"$POETRY_BIN:\$PATH\"" >> ~/.bashrc
    echo "Added Poetry to ~/.bashrc PATH"
fi

echo "=== [3/4] Configuring Poetry System Packages Flag ==="
# CRITICAL: Forces Poetry to use the pre-installed CUDA PyTorch wheels inside the container
poetry config virtualenvs.options.system-site-packages true

echo "=== [4/4] Environment verification ==="
echo "Poetry Version: $(poetry --version)"
echo "Python Version: $(python3 --version)"
echo "PyTorch CUDA Available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"

echo "===================================================="
echo "🎉 Setup Complete! Please run: 'source ~/.bashrc' to activate poetry in your current terminal."
echo "===================================================="