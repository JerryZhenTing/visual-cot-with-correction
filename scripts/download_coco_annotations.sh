#!/bin/bash
# Download COCO 2017 instance annotations (~252 MB).
# We only need the annotation JSON files — NOT the images
# (images are already available via HuggingFace).
#
# Output:
#   data/coco_annotations/instances_train2017.json
#   data/coco_annotations/instances_val2017.json
#
# Run on the Bridges2 login node.
# Usage: bash scripts/download_coco_annotations.sh

set -e

DEST="/jet/home/zliu51/project/data/coco_annotations"
ZIP="/tmp/coco_annotations_$$.zip"

mkdir -p "$DEST"

echo "Downloading COCO 2017 annotations (~252 MB)..."
wget -q --show-progress \
    -O "$ZIP" \
    http://images.cocodataset.org/annotations/annotations_trainval2017.zip

echo "Extracting instance annotation files..."
unzip -j "$ZIP" \
    "annotations/instances_train2017.json" \
    "annotations/instances_val2017.json" \
    -d "$DEST"

rm "$ZIP"

echo ""
echo "Done. Files:"
ls -lh "$DEST"/instances_*.json
echo ""
echo "Next: python src/build_vsr_bboxes.py"
