#!/bin/bash
set -euo pipefail

# Set up directories
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
META_FIGS_DIR="$SCRIPT_DIR/../meta_figs"
CSV_DIR="$SCRIPT_DIR/../csv_files"
HOME_DIR="/home/finn.wimberly"
RCLONE="$HOME_DIR/bin/rclone"


# Sync all whaling figures (exclude .ipynb_checkpoints)
"$RCLONE" sync \
  "$META_FIGS_DIR/" \
  whaling_logbooks_GDrive:project_status/figures \
  --exclude "**/.ipynb_checkpoints/**"

# Copy Tier4logbooks_meta.csv
"$RCLONE" copy \
  "$CSV_DIR/Tier4logbooks_meta.csv" \
  whaling_logbooks_GDrive:project_status/