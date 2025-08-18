from scale import Scale
import pandas as pd
import numpy as np


class Euclidean:
    """
    Compute the Euclidean frame-by-frame distance traveled per player.

    Parameters
    ----------
    csv_path : str
        Path to the .csv files with positions of each player at each frame
    """

    def __init__(self, csv_path: str, schema_points_path: str):
        scale = Scale(schema_points_path)

        self.df = pd.read_csv(csv_path)

        self.df = self.df.sort_values(by=["frame"])

        self.df1 = self.df[self.df['id'] == 1]
        self.df2 = self.df[self.df['id'] == 2]

        # print(self.df1)

        total_distance = 0.0
        for index, row in self.df1.iterrows():
            print(row['x1'], row['y1'], row['x2'], row['y2'])
            x1_center = (row['x1'] + row['x2']) / 2
            y1_center = (row['y1'] + row['y2']) / 2
            if index > 0:
                dist = np.sqrt((x1_center - prev_x1_center) ** 2 + (y1_center - prev_y1_center) ** 2)
                total_distance += dist * scale.scale
            prev_x1_center = x1_center
            prev_y1_center = y1_center
        print(total_distance)