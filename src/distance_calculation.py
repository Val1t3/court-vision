import pandas as pd
import numpy as np


class Euclidean:
    """
    Compute the Euclidean frame-by-frame distance traveled per player
    """
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.df = self.df.sort_values(by=["id", "frame"])

        # # Compute the center point of the bounding box
        # self.df['x_center'] = (self.df['x1'] + self.df['x2']) / 2
        # self.df['y_center'] = (self.df['y1'] + self.df['y2']) / 2

        # Sort by player ID and frame
        self.df = self.df.sort_values(by=['id', 'frame'])

        # Compute previous x and y for each player
        self.df['x_prev'] = self.df.groupby('id')['x'].shift(1)
        self.df['y_prev'] = self.df.groupby('id')['y'].shift(1)

        # Euclidean distance between consecutive frames
        self.df['distance'] = np.sqrt((self.df['x'] - self.df['x_prev'])**2 + (self.df['y'] - self.df['y_prev'])**2)

        # Fill NaN values (first frame) with 0
        self.df['distance'] = self.df['distance'].fillna(0)

        # Total distance per player
        total_distance = self.df.groupby('id')['distance'].sum().reset_index()
        total_distance.columns = ['player_id', 'total_distance']

        # Save results
        # self.df.to_csv("player_distances_per_frame.csv", index=False)
        total_distance.to_csv("saves/euclidean_total_distance.csv", index=False)


if __name__ == "__main__":
    Euclidean(csv_path="saves/point_positions.csv")
