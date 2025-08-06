import pandas as pd
import numpy as np
from filterpy.kalman import KalmanFilter
import cv2
import json


# consts
name = "eval_11"
csv_path = "saves/smoothed_positions_" + name + ".csv"
schema_points_path = "data/points_cropped_schema.json"

class Scale:
    def __init__(self):
        with open(schema_points_path, "r") as f:
            self.points = json.load(f)
            # define points
            top_left = self.points[0]
            top_right = self.points[8]
            bottom_left = self.points[7]
            bottom_right = self.points[15]
            # calculate distances in pixels
            pixels_top = np.linalg.norm(np.array(top_right) - np.array(top_left))
            pixels_bottom = np.linalg.norm(np.array(bottom_right) - np.array(bottom_left))
            pixels_left = np.linalg.norm(np.array(bottom_left) - np.array(top_left))
            pixels_right = np.linalg.norm(np.array(bottom_right) - np.array(top_right))
            # define real distances in meter
            long_meters = 28.0
            short_meters = 15.0
            # calculate scales
            scale_top = long_meters / pixels_top
            scale_bottom = long_meters / pixels_bottom
            scale_left = short_meters / pixels_left
            scale_right = short_meters / pixels_right
        # calculate mean scale
        self.scale = (scale_top + scale_bottom + scale_left + scale_right) / 4


class Euclidean:
    """
    Compute the Euclidean frame-by-frame distance traveled per player.

    Parameters
    ----------
    csv_path : str
        Path to the .csv files with positions of each player at each frame
    """
    def __init__(self, csv_path: str):
        scale = Scale()
        self.df = pd.read_csv(csv_path)
        self.df = self.df.sort_values(by=["id", "frame"])

        # Sort by player ID and frame
        self.df = self.df.sort_values(by=['id', 'frame'])

        # Compute previous x and y for each player
        self.df['x_prev'] = self.df.groupby('id')['x'].shift(1)
        self.df['y_prev'] = self.df.groupby('id')['y'].shift(1)

        # Euclidean distance between consecutive frames
        self.df['distance'] = np.sqrt((self.df['x'] - self.df['x_prev'])**2 + (self.df['y'] - self.df['y_prev'])**2) * scale.scale

        # Fill NaN values (first frame) with 0
        self.df['distance'] = self.df['distance'].fillna(0)

        # Total distance per player
        total_distance = self.df.groupby('id')['distance'].sum().reset_index()
        total_distance.columns = ['player_id', 'total_distance']

        # Save results
        # self.df.to_csv("player_distances_per_frame.csv", index=False)
        total_distance.to_csv("saves/euclidean_total_distance.csv", index=False)


class Kalman:
    """
    Compute the Kalman Filter trajectory smoothing to calculate distance
    traveled per each player.

    Parameters
    ----------
    csv_path : str
        Path to the .csv files with positions of each player at each frame

    """
    def __init__(self, csv_path : str):
        scale = Scale()
        self.df = pd.read_csv(csv_path)
        self.df = self.df.groupby('id').apply(self.smooth_with_kalman, include_groups=False).reset_index()

        # Then compute distance exactly like before:
        self.df['x_smooth_prev'] = self.df.groupby('id')['x_smooth'].shift(1)
        self.df['y_smooth_prev'] = self.df.groupby('id')['y_smooth'].shift(1)
        self.df['distance_kalman'] = np.sqrt((self.df['x_smooth'] - self.df['x_smooth_prev'])**2 +
                                        (self.df['y_smooth'] - self.df['y_smooth_prev'])**2) * scale.scale
        self.df['distance_kalman'] = self.df['distance_kalman'].fillna(0)

        # Total distance per player
        total_distance = self.df.groupby('id')['distance_kalman'].sum().reset_index()
        total_distance.columns = ['player_id', 'total_distance']

        # Save results
        total_distance.to_csv("saves/kalman_total_distance.csv", index=False)


    def apply_kalman_filter(self, x, y):
        kf = KalmanFilter(dim_x=4, dim_z=2)

        # State: [x, y, dx, dy]
        kf.x = np.array([x[0], y[0], 0., 0.])
        kf.F = np.array([[1, 0, 1, 0],
                        [0, 1, 0, 1],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])  # State transition matrix
        kf.H = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0]])  # Measurement matrix
        kf.P *= 1000.  # Covariance matrix
        kf.R = np.eye(2) * 5  # Measurement noise
        kf.Q = np.eye(4)  # Process noise

        filtered = []
        for i in range(len(x)):
            z = np.array([x[i], y[i]])
            kf.predict()
            kf.update(z)
            filtered.append(kf.x[:2].copy())  # Only x, y

        filtered = np.array(filtered)
        return filtered[:, 0], filtered[:, 1]


    def smooth_with_kalman(self, group):
        x_smooth, y_smooth = self.apply_kalman_filter(group['x'].values, group['y'].values)
        group['x_smooth'] = x_smooth
        group['y_smooth'] = y_smooth
        return group


class OpticalFlow:
    def __init__(self, csv_path: str, video_path: str):
        self.df = pd.read_csv(csv_path)
        self.df = self.df.sort_values(by=["id", "frame"]).reset_index(drop=True)

        cap = cv2.VideoCapture(video_path)
        # Parameters for LK optical flow
        lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Convert frames to grayscale for optical flow
        ret, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        # Store last known positions per player
        player_points = {}   # id -> np.array([[x, y]])
        player_distances = {}  # id -> float

        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detections at this frame
            current_detections =self.df[self.df["frame"] == frame_num]

            for _, row in current_detections.iterrows():
                pid = int(row["id"])
                p_current = np.array([[row["x"], row["y"]]], dtype=np.float32)

                if pid in player_points:
                    p_prev = player_points[pid]
                    p_next, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p_prev, None, **lk_params)

                    if st[0][0] == 1:  # Successfully tracked
                        dist = np.linalg.norm(p_next[0] - p_prev[0])
                        player_distances[pid] += dist
                        player_points[pid] = p_next
                    else:
                        player_points[pid] = p_current  # Reset on failure
                else:
                    player_points[pid] = p_current
                    player_distances[pid] = 0.0

            old_gray = gray.copy()
            frame_num += 1

        # Convert to DataFrame
        distance_df = pd.DataFrame(list(player_distances.items()), columns=["player_id", "optical_flow_distance"])
        distance_df.to_csv("saves/optical_flow_distances.csv", index=False)


if __name__ == "__main__":
    Euclidean(csv_path=csv_path)
    Kalman(csv_path=csv_path)
    # OpticalFlow(
    #     csv_path=csv_path,
    #     video_path="assets/easy_1.mov"
    # )