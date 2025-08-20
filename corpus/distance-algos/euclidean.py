from scale import Scale
import pandas as pd
import numpy as np
from baseline_detection import apply_homography


def calculate_distance(points: list, h: np.ndarray, scale: float):
    total_distance = 0.0

    for i in range(len(points) - 1):
        start_pt = points[i]
        end_pt = points[i + 1]

        # apply homography
        new_start_pt = apply_homography(pt=start_pt, h_matrix=h)

        new_end_pt = apply_homography(pt=end_pt, h_matrix=h)

        # calculate distance between points
        distance = np.sqrt(
            (new_end_pt[0] - new_start_pt[0]) ** 2 +
            (new_end_pt[1] - new_start_pt[1]) ** 2
        ) * scale

        total_distance += distance

    return total_distance


class Euclidean:
    """
    Compute the Euclidean frame-by-frame distance traveled per player.

    Parameters
    ----------
    csv_path : str
        Path to the .csv files with positions of each player at each frame
    """

    def __init__(self, csv_path: str, schema_points_path: str, h: np.ndarray):
        scale = Scale(schema_points_path)

        self.df = pd.read_csv(csv_path)

        self.df = self.df.sort_values(by=["frame"])

        self.df1 = self.df[self.df['id'] == 1]
        self.df2 = self.df[self.df['id'] == 2]


        # create list of points
        self.points = []
        for _, row in self.df1.iterrows():
            # convert box to x;y point
            x = row['x1'] + (row['x2'] - row['x1']) / 2
            y = row['y2']

            self.points.append((x, y))

        # calculate distance
        distance = calculate_distance(self.points, h, scale.scale)

        print(f"Total distance for player 1: {distance} meters")
