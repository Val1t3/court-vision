# Tracking Algorithms

A tool for tracking detected objects across video frames using various tracking algorithms.

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Object Tracking

Execute object tracking using available algorithms:

```bash
python3 main.py video.mp4 \
    --out-video out-video.mp4 \
    --out-csv out-file.csv \
    --weights yolo11m.pt \
    --conf 0.25 \
    --iou 0.45
```
Returns analyzed video .mp4 and a .csv file with coordinates of each player on each frame of the video.

> **Info:** You can use the `tracking_pipeline.sh` file to use this function on each file of the `courtvision-dataset` dataset.

### Parameters

| Parameter | Description | Options/Default |
|-----------|-------------|-----------------|
| `video.mp4` | Input video file path | Required |
| `--out-video` | Output video file path | Optional |
| `--out-csv` | Output CSV file path | Optional |
| `--weights` | YOLO model weights file | Default: yolo11m.pt |
| `--conf` | Confidence threshold | Default: 0.25 |
| `--iou` | IoU threshold for NMS | Default: 0.45 |
