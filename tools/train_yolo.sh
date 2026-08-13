#!/bin/sh
# Train the four-leaf detector.
#
# Rotation augmentation is turned all the way up on purpose: a four-leaf is
# 90-degree symmetric and a three-leaf 120-degree, so orientation must never
# become the cue that separates them.
#
# The dataset now carries ~500 unlabelled photographs with no clover in them
# (tools/fetch_bg.py). v1 had none, and pointed at a desk it reported four-leaf
# at 82% on a computer monitor: every training image was a photograph of clover,
# so nothing ever taught it to stay quiet.
set -e
cd "$(dirname "$0")/.."
yolo detect train \
  data=data/yolo/data.yaml \
  model=yolov8n.pt \
  imgsz=640 epochs=120 batch=32 device=0 workers=8 \
  degrees=180 fliplr=0.5 flipud=0.5 \
  hsv_h=0.02 hsv_s=0.5 hsv_v=0.4 \
  scale=0.5 translate=0.15 mosaic=1.0 close_mosaic=15 \
  patience=40 seed=20260813 \
  project=data/yolo_runs name=${1:-v2} exist_ok=True
