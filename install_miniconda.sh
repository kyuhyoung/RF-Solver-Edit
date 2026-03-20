#!/bin/bash

# Install Miniconda3
#
# Usage:
#   ./install_miniconda.sh              # install to ~/miniconda3 (default)
#   ./install_miniconda.sh /custom/path # install to custom path

set -e

INSTALL_DIR="${1:-$HOME/miniconda3}"

echo "Installing Miniconda3 to ${INSTALL_DIR}..."

wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p "${INSTALL_DIR}"
rm /tmp/miniconda.sh

# Initialize conda for bash and fish
"${INSTALL_DIR}/bin/conda" init bash
"${INSTALL_DIR}/bin/conda" init fish 2>/dev/null || true

echo ""
echo "Miniconda3 installed. Restart your shell or run:"
echo "  source ~/.bashrc"
