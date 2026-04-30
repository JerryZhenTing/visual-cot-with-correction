#!/bin/bash
# Download VSR annotation files from the official GitHub repository.
# Annotations contain obj1_bbox and obj2_bbox for each example,
# needed to compute RSA/IoU against ground-truth object regions.
#
# Run on the Bridges2 login node (no GPU needed).
# Usage: bash scripts/download_annotations.sh

set -e

DEST_DIR="/jet/home/zliu51/project/data"
TMP_DIR="/tmp/vsr_repo_$$"

mkdir -p "$DEST_DIR"

echo "Cloning VSR repository (shallow, no history)..."
git clone --depth 1 https://github.com/cambridgeltl/visual-spatial-reasoning "$TMP_DIR"

echo ""
echo "Files found in repo data/ directory:"
find "$TMP_DIR" -name "*.json" | sort

# Try common annotation file locations in order
FOUND=0
for candidate in \
    "$TMP_DIR/data/zeroshot/dev.json" \
    "$TMP_DIR/data/zeroshot/val.json" \
    "$TMP_DIR/data/dev.json" \
    "$TMP_DIR/data/val.json" \
    "$TMP_DIR/data/zeroshot/test.json"; do
    if [ -f "$candidate" ]; then
        cp "$candidate" "$DEST_DIR/vsr_annotations.json"
        echo ""
        echo "Copied: $candidate → $DEST_DIR/vsr_annotations.json"
        FOUND=1
        break
    fi
done

rm -rf "$TMP_DIR"

if [ "$FOUND" -eq 0 ]; then
    echo ""
    echo "ERROR: Could not find a dev/val annotation file automatically."
    echo "Check the file list above and copy the right file manually:"
    echo "  cp <path_from_above> $DEST_DIR/vsr_annotations.json"
    exit 1
fi

# Print first entry so user can verify the format
echo ""
echo "First annotation entry (for format verification):"
python -c "
import json
with open('$DEST_DIR/vsr_annotations.json') as f:
    data = json.load(f)
print(json.dumps(data[0], indent=2))
print(f'Total entries: {len(data)}')
"

echo ""
echo "Done. load_vsr.py will pick up the annotations automatically."
