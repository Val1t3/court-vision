import numpy as np
import json


class Scale:
    def __init__(self, schema_points_path: str):
        with open(schema_points_path, "r") as f:
            self.points = json.load(f)

        # define points
        pt_0 = self.points[0]
        pt_8 = self.points[8]
        pt_7 = self.points[7]
        pt_15 = self.points[15]

        # calculate distances in pixels
        pixels_top = np.sqrt(
            (pt_8[0] - pt_0[0]) ** 2 + (pt_8[1] - pt_0[1]) ** 2
        )
        pixels_bottom = np.sqrt(
            (pt_15[0] - pt_7[0]) ** 2 + (pt_15[1] - pt_7[1]) ** 2
        )
        pixels_left = np.sqrt(
            (pt_7[0] - pt_0[0]) ** 2 + (pt_7[1] - pt_0[1]) ** 2
        )
        pixels_right = np.sqrt(
            (pt_15[0] - pt_8[0]) ** 2 + (pt_15[1] - pt_8[1]) ** 2
        )

        # define real distances in meter
        long_meters = 28.0
        short_meters = 15.0

        # calculate scales (meters per pixel)
        scale_top = long_meters / pixels_top
        scale_bottom = long_meters / pixels_bottom
        scale_left = short_meters / pixels_left
        scale_right = short_meters / pixels_right

        # calculate mean scale (meters per pixel)
        self.scale = (scale_top + scale_bottom + scale_left + scale_right) / 4
