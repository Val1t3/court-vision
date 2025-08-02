# Court Vision
Valentin Woehrel - 2025

## Architecture

This project is organized into the following directories:

- **assets/** — Contains videos, extracted first frames, and schema illustrations
- **data/** — Stores coordinates of key frame points and schema reference data
- **models/** — Pre-trained models used for player detection
- **other/** — Miscellaneous Python scripts used for testing or experimentation
- **output/** — Output videos with player detection overlays
- **saves/** — Detected player coordinates saved during processing

---
## Install Dependencies
```bash
pip install -r requirements.txt
```

---
## First Frame

Use this script to extract the first frame of your video.

```bash
python first_frame_video.py "video_path"
```
---
## Point selection

Use this tiny tool to select the main points of the court on the first frame.

![Court schema points](../assets/schema_points.png)

```bash
python point_select.py "frame_path" "point_path"
```

---
## Run

---

