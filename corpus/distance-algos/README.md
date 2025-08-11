# Distance Algorithms

A tool for calculating distances between detected objects using various algorithms.

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Distance Calculation

Execute distance calculations using all available algorithms:

```bash
python3 main.py \
    --csv detections.csv \
    --img-points image_points.json \
    --court-points court_points.json \
    --video video.mp4 \
    --fps 25.0 \
    --feet-point bottom_center \
    --court-units schema_px \
    --schema-length-m 28.0 \
    --schema-width-m 15.0
```

### Parameters

| Parameter | Description | Options/Default |
|-----------|-------------|-----------------|
| `--csv` | Detection results file | Required |
| `--img-points` | Image coordinate points | Required |
| `--court-points` | Court reference points | Required |
| `--video` | Video file path | Required for optical flow |
| `--fps` | Frames per second | Default: 25.0 (used if --video omitted) |
| `--feet-point` | Point selection method | `bottom_center`, `bottom_mid`, `feet`, `center` |
| `--court-units` | Unit system for court measurements | `meters`, `schema_px` |
| `--schema-length-m` | Physical court length (x-axis) | Default: 28.0 meters (when court-units=schema_px) |
| `--schema-width-m` | Physical court width (y-axis) | Default: 15.0 meters (when court-units=schema_px) |