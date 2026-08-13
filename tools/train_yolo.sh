#!/bin/sh
# Train the four-leaf detector. Rotation augmentation is turned all the way up
# on purpose: a four-leaf is 90-degree symmetric and a three-leaf 120-degree,
# so orientation must never become the cue that separates them.
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
  project=data/yolo_runs name=v1 exist_ok=True
