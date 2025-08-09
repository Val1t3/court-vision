import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
from scale import Scale


class Kalman:
    """
    Compute the Kalman Filter trajectory smoothing to calculate distance
    traveled per each player.

    Parameters
    ----------
    csv_path : str
        Path to the .csv files with positions of each player at each frame

    """
    def __init__(self, csv_path: str, schema_points_path: str):
        scale = Scale(schema_points_path)

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
        total_distance.to_csv("../data/saves/kalman_total_distance.csv", index=False)


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
