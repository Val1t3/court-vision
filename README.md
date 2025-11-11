## Court Vision

Modular video analysis pipeline to extract player statistics from amateur basketball footage. The system is engineered to overcome real-world challenges like low-quality video, variable camera angles, and detection noise.

### Abstract

This dissertation presents a modular pipeline for estimating player movement in amateur basketball videos, comparing three key distance estimation algorithms: Euclidean frame-based distance, Kalman filtering, and optical flow. The system is tailored for real-world challenges such as low-quality footage, variable camera angles, and detection noise.

The workflow integrates automated player detection (YOLOv8), robust tracking (ByteTrack), manual homography calibration with a custom tool, and applies all three algorithms to a standardized dataset containing synchronized videos, ground truth distances, and tracked player positions. This design allows for direct and fair algorithm comparisons.

Evaluation includes both qualitative visualization of player trajectories and quantitative benchmarking against ground truth. Results indicate that the Euclidean method, while fast, is overly sensitive to noise and often overestimates distances. The Kalman filter offers the best balance of accuracy and efficiency, making it suitable for most applications. Optical flow can outperform Kalman in some cases with unreliable detections but is hindered by much higher computational costs.

The open-source framework and dataset promote further research and development. Future improvements will focus on automating calibration, expanding the dataset, adding new metrics, and refining algorithms for greater robustness.

### Repository Structure

The repository is organized into two main sections:

- `research/` — Experimental, deprecated, or exploratory code and assets from prototyping and investigations. See `research/README.md` for context and references.
- `corpus/` — The official corpus used in the thesis: standardized data, assets, scripts, and pipelines for detection, tracking, calibration, and distance estimation. See `corpus/README.md` for details.

Quick links:

- [`research/`](research/README.md)
- [`corpus/`](corpus/README.md)

### Pipeline Overview

- **Detection**: YOLOv8-based player detection.
- **Tracking**: ByteTrack for robust multi-object tracking.
- **Calibration**: Manual homography estimation (custom point-selection tool) to map image to court coordinates.
- **Distance Estimation**: Three interchangeable algorithms:
  - Euclidean frame-wise distance
  - Kalman filter–based smoothing and distance
  - Optical flow–based motion estimation
- **Evaluation**:
  - Qualitative: trajectory visualization and overlays
  - Quantitative: benchmarking vs. ground-truth distances
