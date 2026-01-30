#!/usr/bin/env bash
set -euo pipefail

# Change these paths according to your actual directory structure
BASE_PATH="content/object-tracking"
MAIN_PY_SOURCE="/${BASE_PATH}/main.py"
SAM2_ROOT="/${BASE_PATH}/sam2"
CHECKPOINT_SOURCE="/content/sam2.1_hiera_tiny.pt"
CHECKPOINT_DEST="${SAM2_ROOT}/checkpoints"

# ===============================================

echo "┌───────────────────────────────────────────────┐"
echo "│          Object Tracking Setup & Launch         │"
echo "└───────────────────────────────────────────────┘"
echo ""

# 1. Copy main.py
echo "→ Copying main.py ..."
if [ ! -f "$MAIN_PY_SOURCE" ]; then
    echo "Error: main.py not found at $MAIN_PY_SOURCE"
    exit 1
fi
cp -v "$MAIN_PY_SOURCE" "$SAM2_ROOT/"
echo ""

# 2. Copy checkpoint
echo "→ Copying checkpoint file(s) ..."
if [ ! -e "$CHECKPOINT_SOURCE" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT_SOURCE"
    exit 1
fi

mkdir -p "$CHECKPOINT_DEST"
if [ -f "$CHECKPOINT_SOURCE" ]; then
    # single file
    cp -v "$CHECKPOINT_SOURCE" "$CHECKPOINT_DEST/"
elif [ -d "$CHECKPOINT_SOURCE" ]; then
    # directory → copy contents
    cp -v -r "$CHECKPOINT_SOURCE"/* "$CHECKPOINT_DEST/"
else
    echo "Error: $CHECKPOINT_SOURCE is neither file nor directory"
    exit 1
fi
echo ""

# 3. Go to sam2 directory
echo "→ Changing directory for installing required libraries"
cd "$SAM2_ROOT" || { echo "Cannot cd to $SAM2_ROOT"; exit 1; }
pwd
echo ""


# 4. Install required libraries in editable mode
echo "→ Installing libraries in editable mode (-e) ..."
pip install --upgrade gradio
pip install -e .
echo "build finished"

