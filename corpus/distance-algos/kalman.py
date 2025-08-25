import numpy as np
import pandas as pd
from scale import Scale
from baseline_detection import apply_homography


class KalmanFilter:
    """
    2D Kalman filter for tracking player position and velocity.
    """

    def __init__(self, dt=1.0, process_noise=1.0, measurement_noise=1.0):
        self.dt = dt  # time step

        # State vector: [x, y, vx, vy] (position and velocity)
        self.x = np.zeros((4, 1))

        # State transition matrix
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Measurement matrix (we only observe position)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Process noise covariance
        self.Q = process_noise * np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2]
        ])

        # Measurement noise covariance
        self.R = measurement_noise * np.eye(2)

        # Error covariance matrix
        self.P = np.eye(4) * 1000

    def predict(self):
        """Predict the next state."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement):
        """Update state with new measurement."""
        y = measurement.reshape(-1, 1) - self.H @ self.x  # residual
        S = self.H @ self.P @ self.H.T + self.R  # residual covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def get_position(self):
        """Get current position estimate."""
        return self.x[0, 0], self.x[1, 0]


def calculate_kalman_distance(points: list, h: np.ndarray, scale: float,
                            process_noise=1.0, measurement_noise=10.0):
    """
    Calculate distance using Kalman filter for smoother position estimates.

    Parameters
    ----------
    points : list
        List of (x, y) position tuples
    h : np.ndarray
        Homography matrix
    scale : float
        Scale factor for distance conversion
    process_noise : float
        Process noise parameter for Kalman filter
    measurement_noise : float
        Measurement noise parameter for Kalman filter
    """
    if len(points) < 2:
        return 0.0

    # Initialize Kalman filter
    kf = KalmanFilter(
        dt=1.0,
        process_noise=process_noise,
        measurement_noise=measurement_noise
    )

    # Initialize with first point (after homography)
    first_point = apply_homography(pt=points[0], h_matrix=h)
    kf.x[0, 0] = first_point[0]  # x position
    kf.x[1, 0] = first_point[1]  # y position

    total_distance = 0.0
    step_distances = [0.0]
    cum_distances = [0.0]
    prev_filtered_pos = first_point

    for i in range(1, len(points)):
        # Apply homography to current measurement
        current_measurement = apply_homography(pt=points[i], h_matrix=h)

        # Predict next state
        kf.predict()

        # Update with measurement
        kf.update(np.array(current_measurement))

        # Get filtered position
        filtered_pos = kf.get_position()

        # Calculate distance between filtered positions
        distance = np.sqrt(
            (filtered_pos[0] - prev_filtered_pos[0]) ** 2 +
            (filtered_pos[1] - prev_filtered_pos[1]) ** 2
        ) * scale

        total_distance += distance
        step_distances.append(distance)
        cum_distances.append(total_distance)
        prev_filtered_pos = filtered_pos

    return total_distance, step_distances, cum_distances


class KalmanDistance:
    """
    Compute the Kalman filter-based frame-by-frame distance traveled per player.

    Parameters
    ----------
    csv_path : str
        Path to the .csv files with positions of each player at each frame
    schema_points_path : str
        Path to schema points for scale calculation
    h : np.ndarray
        Homography matrix
    process_noise : float, optional
        Process noise parameter for Kalman filter (default: 1.0)
    measurement_noise : float, optional
        Measurement noise parameter for Kalman filter (default: 10.0)
    """

    def __init__(self, csv_path: str, schema_points_path: str, h: np.ndarray,
                 output_csv_path: str, process_noise: float = 1.0,
                 measurement_noise: float = 10.0):
        scale = Scale(schema_points_path)

        self.df = pd.read_csv(csv_path)
        self.df = self.df.sort_values(by=["frame"])

        self.df1 = self.df[self.df['id'] == 1]
        self.df2 = self.df[self.df['id'] == 2]

        # Process player 1
        self.points1 = []
        frames = []
        for _, row in self.df1.iterrows():
            # convert box to x,y point
            x = row['x1'] + (row['x2'] - row['x1']) / 2
            y = row['y2']
            self.points1.append((x, y))
            frames.append(row["frame"])

        # Calculate distances using Kalman filter
        distance1, step_distances, cum_distances = calculate_kalman_distance(
            self.points1, h, scale.scale, process_noise, measurement_noise
        )

        results = pd.DataFrame({
            'frame': frames,
            'id': 1,
            'step_m': step_distances,
            'cum_m': cum_distances,
            'method': 'kalman'
        })

        results.to_csv(output_csv_path, index=False)
        print(f"Results written to {output_csv_path}")
        print(f"Kalman distance: {distance1:.2f} meters")