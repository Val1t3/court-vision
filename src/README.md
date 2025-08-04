# Court Vision
Valentin Woehrel - 2025

## Architecture

This project is organized into the following directories:

```bash
  .
  ├── assets/  — Contains videos, extracted first frames, and schema illustrations
  ├── data/    — Stores coordinates of key frame points and schema reference data
  ├── models/  — Pre-trained models used for player detection
  ├── other/   — Miscellaneous Python scripts used for testing or experimentation
  ├── output/  — Output videos with player detection overlays
  └── saves/   — Detected player coordinates saved during processing
```


## Install Dependencies
```bash
pip install -r requirements.txt
```


## First Frame

This script save the first frame of a video:
```bash
python first_frame_video.py "video_path"
```

## Point selection

Use this interactive tool to select key court points on the first frame:
```bash
python point_select.py "frame_path" "point_path"

or

python point_select.py "frame_path" "point_path" "point_index"  -> modify the given point only
```

![Court schema points](../assets/schema_points.png)


## Run

### Show Homography

Display the application of homography in real time.

> 📌 Note: You can change the input video path directly in the script.

```bash
python court_line_video_fix.py
```

![Court schema points](../assets/show_homography.png)
