# #!/bin/bash
# set -euo pipefail

# # extract export base dir
# SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# BASE="$(basename "$SCRIPT_DIR")"
# STAMP="${BASE%%_*}"

# # extract YMD parts
# YEAR="${STAMP:2:2}"
# MONTH="${STAMP:4:2}"
# DAY="${STAMP:6:2}"

# # local paths
# HOME_DIR="/home/finn.wimberly"
# CSV_DIR="$SCRIPT_DIR/../csv_files"
# PKL_DIR="$SCRIPT_DIR/../pkl_files"

# # build folder structure for drive
# REMOTE_ROOT="whaling_logbooks_GDrive:data/exports"
# REMOTE="${REMOTE_ROOT}/${MONTH}-${DAY}-${YEAR}_export"

#!/bin/bash
set -euo pipefail

# 1. Get the directory of the script (utils folder)
# Using 'cd' ensures we have an absolute path regardless of where you call it from
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 2. Get the parent directory (the 20260211_export folder)
EXPORT_DIR="$(dirname "$SCRIPT_DIR")"
BASE=$(basename "$EXPORT_DIR")

# 3. Extract the 8-digit date (e.g., 20260211)
STAMP="${BASE%%_*}"

# 4. Safety Check
if [[ ! $STAMP =~ ^[0-9]{8}$ ]]; then
    echo "Error: Could not parse an 8-digit date from: '$BASE'"
    exit 1
fi

# 5. Extract YMD parts
YEAR="${STAMP:0:4}"
MONTH="${STAMP:4:2}"
DAY="${STAMP:6:2}"

# 6. LOCAL PATHS - These are now inside the export folder
HOME_DIR="/home/finn.wimberly"
CSV_DIR="$EXPORT_DIR/csv_files"
PKL_DIR="$EXPORT_DIR/pkl_files"

# 7. REMOTE PATHS
REMOTE_ROOT="whaling_logbooks_GDrive:data/exports"
REMOTE="${REMOTE_ROOT}/${MONTH}-${DAY}-${YEAR}_export"

echo "Syncing from: $EXPORT_DIR"
echo "Targeting Remote: $REMOTE"

# 8. Sync with rclone
"$HOME_DIR/bin/rclone" copy "$CSV_DIR" "$REMOTE/csv_files" \
  --filter "- .ipynb_checkpoints/**" \
  --filter "+ Tier*.csv" \
  --filter "- *"

"$HOME_DIR/bin/rclone" copy "$PKL_DIR" "$REMOTE/pkl_files" \
  --filter "- .ipynb_checkpoints/**" \
  --filter "+ Tier*.pkl" \
  --filter "- *"

echo "Sync Complete."