#!/usr/bin/env bash

BASE_PATH="content/object-tracking"
SAM2_ROOT="/${BASE_PATH}/sam2"

# 1. Go To SAM2 folder
echo "→ Changing directory for installing required libraries"
cd "$SAM2_ROOT" || { echo "Cannot cd to $SAM2_ROOT"; exit 1; }
pwd
echo ""

# 2. Launch
echo "→ Starting the application..."
echo ""

python3 main.py

echo ""