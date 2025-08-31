import pandas as pd
import numpy as np
from scale import Scale
from baseline_detection import apply_homography
import time


def calculate_step_distances(points: list, h: np.ndarray, scale: float):
    step_distances = [0.0]  # First frame has 0 step distance
    cum_distances = [0.0]
    total_distance = 0.0

    for i in range(1, len(points)):
        start_pt = points[i - 1]
        end_pt = points[i]

        new_start_pt = apply_homography(pt=start_pt, h_matrix=h)
        new_end_pt = apply_homography(pt=end_pt, h_matrix=h)

        distance = np.sqrt(
            (new_end_pt[0] - new_start_pt[0]) ** 2 +
            (new_end_pt[1] - new_start_pt[1]) ** 2
        ) * scale

        total_distance += distance
        step_distances.append(distance)
        cum_distances.append(total_distance)

    return total_distance, step_distances, cum_distances


class Euclidean:
    def __init__(self, csv_path: str, schema_points_path: str, h: np.ndarray, output_csv_path: str):
        start_time = time.time()
        scale = Scale(schema_points_path)
        self.df = pd.read_csv(csv_path)
        self.df = self.df.sort_values(by=["frame"])
        self.df1 = self.df[self.df['id'] == 1]

        self.points = []
        frames = []
        for _, row in self.df1.iterrows():
            x = row['x1'] + (row['x2'] - row['x1']) / 2
            y = row['y2']
            self.points.append((x, y))
            frames.append(row['frame'])

        total_distance, step_distances, cum_distances = calculate_step_distances(self.points, h, scale.scale)

        results = pd.DataFrame({
            'frame': frames,
            'id': 1,
            'step_m': step_distances,
            'cum_m': cum_distances,
            'method': 'euclidean'
        })

        results.to_csv(output_csv_path, index=False)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Results written to {output_csv_path}")
        print(f"Euclidean distance: {total_distance} meters")
        print(f"Computing time: {elapsed_time:.4f} seconds")
