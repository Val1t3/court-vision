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
        total_distance.to_csv("../data/saves/euclidean_total_distance.csv", index=False)
