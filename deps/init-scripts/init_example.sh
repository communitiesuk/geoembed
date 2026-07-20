#!/bin/bash
set -e

# install & configure uv
echo "Downloading uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh

echo "Adding uv to PATH"
source $HOME/.local/bin/env

# set the local & UC python wheel variables
WHEEL_NAME="geoembed-0.1.0.dev0-py3-none-any.whl"
SCRIPT_DIR=$(dirname "$0")
LOCAL_WHEEL_PATH="${SCRIPT_DIR}/${WHEEL_NAME}"

UC_WHEEL_PATH="/Volumes/your_unity_catalog/your_schema/whl/${WHEEL_NAME}"

# Extras to install:
# - spatial,cog,stac: always needed (metadata, COG conversion, chipping)
# - models: only needed on GPU clusters (torch/torchgeo for embeddings)
#   Omitted by default to avoid conflicts with DBR's pre-installed torch/sympy.
#   On GPU clusters, torch is already installed by the runtime — torchgeo can be
#   added via %pip in the notebook or a separate GPU init script.
EXTRAS="spatial,cog,stac"

# try local path - assumes init script and wheel exist in the same path
echo "Checking for wheel locally at: ${LOCAL_WHEEL_PATH}"
if [ -f "${LOCAL_WHEEL_PATH}" ]; then
  echo "Wheel file found in script directory."
  echo "Installing wheel with extras: [${EXTRAS}]"
  uv pip install "${LOCAL_WHEEL_PATH}[${EXTRAS}]"
  echo "Installation complete."

# if that fails, try Unity Catalog path
elif [ -f "${UC_WHEEL_PATH}" ]; then
  echo "Local wheel not found. Trying Unity Catalog Volume path: ${UC_WHEEL_PATH}"
  echo "Wheel file found in Unity Catalog Volume."
  echo "Installing wheel with extras: [${EXTRAS}]"
  uv pip install "${UC_WHEEL_PATH}[${EXTRAS}]"
  echo "Installation complete."

else
  echo "Error: Wheel file not found in either location."
  echo "  - Local directory (attempted): ${LOCAL_WHEEL_PATH}"
  echo "  - UC Volume (attempted):       ${UC_WHEEL_PATH}"
  echo ""
  echo "Please ensure '${WHEEL_NAME}' is in the script directory"
  echo "OR that the 'UC_WHEEL_PATH' variable is set correctly in this script."
  exit 1
fi
