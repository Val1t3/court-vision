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

        # pt_16 = self.points[16]
        # pt_17 = self.points[17]

        # pt_2 = self.points[2]
        # pt_3 = self.points[3]
        # pt_4 = self.points[4]
        # pt_5 = self.points[5]
        # pt_10 = self.points[10]
        # pt_11 = self.points[11]
        # pt_12 = self.points[12]
        # pt_13 = self.points[13]

        # calculate distances in pixels
        pixels_top = np.linalg.norm(np.array(pt_8) - np.array(pt_0))
        pixels_bottom = np.linalg.norm(np.array(pt_15) - np.array(pt_7))
        pixels_left = np.linalg.norm(np.array(pt_7) - np.array(pt_0))
        pixels_right = np.linalg.norm(np.array(pt_15) - np.array(pt_8))

        # pixels_mid = np.linalg.norm(np.array(pt_17) - np.array(pt_16))

        # pixels_raq_1 = np.linalg.norm(np.array(pt_3) - np.array(pt_2))
        # pixels_raq_2 = np.linalg.norm(np.array(pt_4) - np.array(pt_5))
        # pixels_raq_3 = np.linalg.norm(np.array(pt_10) - np.array(pt_11))
        # pixels_raq_4 = np.linalg.norm(np.array(pt_13) - np.array(pt_12))

        # pixels_raq_5 = np.linalg.norm(np.array(pt_4) - np.array(pt_3))
        # pixels_raq_6 = np.linalg.norm(np.array(pt_12) - np.array(pt_11))

        # define real distances in meter
        long_meters = 28.0
        short_meters = 15.0

        long_raq = 5.79
        short_raq = 4.88

        # calculate scales (meters per pixel)
        scale_top = long_meters / pixels_top
        scale_bottom = long_meters / pixels_bottom
        scale_left = short_meters / pixels_left
        scale_right = short_meters / pixels_right

        # scale_mid = short_meters / pixels_mid

        # long_raq_1 = long_raq / pixels_raq_1
        # long_raq_2 = long_raq / pixels_raq_2
        # long_raq_3 = long_raq / pixels_raq_3
        # long_raq_4 = long_raq / pixels_raq_4

        # short_raq_1 = short_raq / pixels_raq_5
        # short_raq_2 = short_raq / pixels_raq_6

        # calculate mean scale (meters per pixel)
        self.scale = (scale_top + scale_bottom + scale_left + scale_right
                    #   + scale_mid
                    #   + long_raq_1 + long_raq_2 + long_raq_3 + long_raq_4
                    #   + short_raq_1 + short_raq_2
                      ) / 4
