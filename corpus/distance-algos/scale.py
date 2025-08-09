import numpy as np
import json


class Scale:
    def __init__(self, schema_points_path: str):
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
