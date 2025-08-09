# Dataset

An evaluation dataset has been developed by Valentin Woehrel to assess the accuracy and performance of algorithms for calculating the distance traveled by basketball players. It includes:

- Video recordings of player movements
- Approximate player positions for each frame
- Ground truth distances traveled (in meters)

```
  .
  ├── eval_1.mov            — Video of the movement
  ├── eval_1_positions.csv  — Coordinates of each player on each frame
  ├── eval_1_distance.csv   — Distance traveled in real life
    ...
```

📁 [Access the dataset on Google Drive](https://drive.google.com/drive/folders/1TxdPJCvVeL86invfrzP5ipZW9437LR8l?usp=sharing)

## Runs

### ‼️ Install Dependencies
```bash
pip install -r requirements.txt
```

### Show Homography

Display the application of homography in real time.

> **Note**: You can change the input video path directly in the script.

```bash
python court_line_video_fix.py
```

![Court schema points](../../assets/show_homography.png)

### Player detection

This script detects and tracks players in a video, then exports their coordinates (per frame) to a `.csv` file for further analysis.

> **Note**: You can configure the input and output paths directly in the script.

```bash
python main.py
```
