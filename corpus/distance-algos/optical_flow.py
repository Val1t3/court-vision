import numpy as np
import pandas as pd
import cv2
from scale import Scale



class OpticalFlow:
    """
    Compute the distance traveled using Optical Flow tracking between frames.
    Includes preprocessing of YOLO bounding boxes to extract center positions.
    """

    def __init__(self, csv_path: str, video_path: str, schema_points_path: str):
        scale = Scale(schema_points_path)

        # Preprocess YOLO CSV to extract center points
        self.df = self.preprocess_yolo_csv(csv_path)

        cap = cv2.VideoCapture(video_path)

        # Parameters for Lucas-Kanade Optical Flow
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Read first frame
        ret, old_frame = cap.read()
        if not ret:
            raise RuntimeError("Unable to read the first frame of the video.")

        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        # Store last known position per player
        player_points = {}      # id -> np.array([[x, y]])
        player_distances = {}   # id -> float

        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Get detections at current frame
            current_detections = self.df[self.df["frame"] == frame_num]

            for _, row in current_detections.iterrows():
                pid = int(row["id"])
                p_current = np.array([[row["x"], row["y"]]], dtype=np.float32)

                if pid in player_points:
                    p_prev = player_points[pid]
                    p_next, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p_prev, None, **lk_params)

                    if st[0][0] == 1:  # Successfully tracked
                        dist_px = np.linalg.norm(p_next[0] - p_prev[0])
                        dist_m = dist_px * scale.scale
                        player_distances[pid] += dist_m
                        player_points[pid] = p_next
                    else:
                        player_points[pid] = p_current  # Reset on failure
                else:
                    player_points[pid] = p_current
                    player_distances[pid] = 0.0

            old_gray = gray.copy()
            frame_num += 1

        # Convert to DataFrame
        distance_df = pd.DataFrame(list(player_distances.items()), columns=["player_id", "total_distance"])
        distance_df.to_csv("../data/saves/optical_flow_total_distance.csv", index=False)

    def preprocess_yolo_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Extracts the center of YOLO bounding boxes for use in optical flow tracking.

        Parameters
        ----------
        csv_path : str
            Path to the CSV file with YOLO format: frame, id, x1, y1, x2, y2, confidence, class

        Returns
        -------
        DataFrame with columns: frame, id, x, y, confidence, class
        """
        df = pd.read_csv(csv_path)
        df["x"] = (df["x1"] + df["x2"]) / 2
        df["y"] = (df["y1"] + df["y2"]) / 2
        return df[["frame", "id", "x", "y", "confidence", "class"]].copy()
