# court-vision/src, baseline_detection.py
# Code written by Valentin Woehrel, 2025

import json
import numpy as np
import cv2
import statistics


def warp_picture(h: np.ndarray, src: np.ndarray, dest: np.ndarray):
    """
    Warp the frame using the homography matrix.

    Returns
    -------
    np.ndarray
        The warped frame.
    """

    # Get the dimensions of the dest
    h_dest, w_dest = dest.shape[:2]

    # Warp the frame using the homography matrix
    warped_res = cv2.warpPerspective(src, h, (w_dest, h_dest))

    return warped_res


def draw_line_between_points(image: np.ndarray, point1: np.ndarray, point2: np.ndarray):
    """
    Draw a line between two points on the image.

    Parameters
    ----------
    image : np.ndarray
        The image on which to draw the line.
    point1 : list
        The first point (x, y).
    point2 : list
        The second point (x, y).
    """

    # Convert points to integers
    point1 = tuple(map(int, point1))
    point2 = tuple(map(int, point2))

    # Draw line on the image
    cv2.line(image, point1, point2, (0, 255, 0), 1)


# def generate_points_on_line(point1: np.ndarray, point2: np.ndarray, num_points: int = 100) -> np.ndarray:
#     """
#     DEPRECATED:
#     Generate `num_points` equally spaced points between point1 and point2.
#
#     Parameters
#     ----------
#     point1 : np.ndarray
#         Starting point (x, y).
#     point2 : np.ndarray
#         Ending point (x, y).
#     num_points : int
#         Number of points to generate.
#
#     Returns
#     -------
#     np.ndarray
#         Array of shape (num_points, 2) containing interpolated points.
#     """
#     point1 = np.array(point1, dtype=np.float32)
#     point2 = np.array(point2, dtype=np.float32)
#
#     return np.linspace(point1, point2, num=num_points)


class BaselineDetection:
    """
    A class used to detect baseline of basket court.

    Attributes
    ----------
    frame : np.ndarray
        Frame image object.
    schema : np.ndarray
        Schema image object.
    frame_points : list
        Points of frame.
    schema_points : list
        Points of schema.
    """

    def __init__(
        self,
        schema: np.ndarray = None,
        frame: np.ndarray = None,
        frame_points_path: str = None,
        schema_points_path: str = None,
    ):
        """
        Initialize the BaselineDetection class.

        Parameters
        ----------
        schema : np.ndarray
            Schema image object.
        frame : np.ndarray
            Frame image object.
        frame_points_path : str
            Path to the frame points file.
        schema_points_path : str
            Path to the schema points file.
        """
        self.frame = frame
        self.schema = schema
        self.tracking_points = []

        # Load frames
        # if self.schema is None:
        #     raise ValueError("[BaselineDetection error]: couldn't load schema")

        # if self.frame is None:
        #     raise ValueError("[BaselineDetection error]: couldn't generate frame")

        # Load points from JSON files
        with open(frame_points_path, "r") as f:
            frame_data = json.load(f)
        with open(schema_points_path, "r") as f:
            schema_data = json.load(f)

        if schema_data is None:
            raise ValueError("[BaselineDetection error]: couldn't load schema points")
        if frame_data is None:
            raise ValueError("[BaselineDetection error]: couldn't load frame points")

        self.frame_points = np.array(frame_data, dtype=np.float32)
        self.schema_points = np.array(schema_data, dtype=np.float32)

    def calculate_homography(self) -> tuple:
        """
        Calculate the homography matrix from the frame points to the schema points.

        Returns
        -------
        np.ndarray
            The homography matrix.
        np.ndarray
            The inverse homography matrix.
        """

        # Check if points are in the correct format
        if len(self.frame_points) < 4 or len(self.schema_points) < 4:
            raise ValueError(
                "[BaselineDetection error]: not enough points to calculate homography"
            )

        # Calculate homography matrix
        h, _ = cv2.findHomography(self.frame_points, self.schema_points)
        h_inv, _ = cv2.findHomography(self.schema_points, self.frame_points)

        # Check if homography matrix is valid
        if h is None or h_inv is None:
            raise ValueError(
                "[BaselineDetection error]: couldn't calculate homography matrix"
            )

        return h, h_inv

    def line_identification(self, warped_img: np.ndarray, court_side: str):
        """
        DEPRECATED:
        Identify lines in the warped image.

        Parameters
        ----------
        warped_img : np.ndarray
            The warped image.
        court_side: int
            The left of right side of the court: left or right

        Returns
        -------
        np.ndarray
            The image with identified lines.
        """

        # Sideline
        draw_line_between_points(
            warped_img, self.schema_points[0], self.schema_points[1]
        )
        draw_line_between_points(
            warped_img, self.schema_points[1], self.schema_points[2]
        )
        draw_line_between_points(
            warped_img, self.schema_points[2], self.schema_points[3]
        )
        draw_line_between_points(
            warped_img, self.schema_points[3], self.schema_points[0]
        )

        # 3-pts Line
        draw_line_between_points(
            warped_img, self.schema_points[4], self.schema_points[5]
        )
        draw_line_between_points(
            warped_img, self.schema_points[6], self.schema_points[7]
        )
        # Half Circle
        center_1 = (
            int((self.schema_points[4][0] + self.schema_points[7][0]) / 2),
            int((self.schema_points[4][1] + self.schema_points[7][1]) / 2),
        )  # Center between points 5 and 6
        center = (
            int((center_1[0] + self.schema_points[1][0]) / 2),
            int(center_1[1]),
        )  # Center of the basket position, between axis x of center_1 and point 2

        radius = int(
            np.linalg.norm(
                np.array(self.schema_points[4]) - np.array(self.schema_points[7])
            )
            / 2
        )
        radius = radius + 5  # Need explanations for the small offset incrementation

        if court_side == "left":
            cv2.ellipse(
                warped_img, center, (radius, radius), -100, 20, 180, (0, 255, 0), 1
            )
        elif court_side == "right":
            cv2.ellipse(
                warped_img, center, (radius, radius), 100, 0, 160, (0, 255, 0), 1
            )
        else:
            raise ValueError(
                "[BaselineDetection error]: bad value for court_side parameter of line_identification function"
            )

        # Lane Line
        draw_line_between_points(
            warped_img, self.schema_points[8], self.schema_points[9]
        )
        draw_line_between_points(
            warped_img, self.schema_points[10], self.schema_points[11]
        )
        draw_line_between_points(
            warped_img, self.schema_points[11], self.schema_points[8]
        )

    def line_identification_full_court(self, warped_img: np.ndarray) -> np.ndarray:
        """
        Identify lines of the full court in the warped image.

        Parameters
        ----------
        warped_img : np.ndarray
            The warped image.

        Returns
        -------
        np.ndarray
            The image with identified lines.
        """

        # Generate missing points
        sections = [
            self.schema_points[3][0] - self.schema_points[2][0],
            self.schema_points[5][0] - self.schema_points[4][0],
            self.schema_points[10][0] - self.schema_points[11][0],
            self.schema_points[12][0] - self.schema_points[13][0],
        ]

        mean_sec = statistics.mean(sections)
        pix_distance = mean_sec / 5.6
        new_point_pix = pix_distance * 2.9
        new_bask_pix = pix_distance * 1.2

        np_18 = [self.schema_points[1][0] + new_point_pix, self.schema_points[1][1]]
        self.schema_points = np.vstack([self.schema_points, np_18])

        np_19 = [self.schema_points[6][0] + new_point_pix, self.schema_points[6][1]]
        self.schema_points = np.vstack([self.schema_points, np_19])

        np_20 = [self.schema_points[9][0] - new_point_pix, self.schema_points[9][1]]
        self.schema_points = np.vstack([self.schema_points, np_20])

        np_21 = [self.schema_points[14][0] - new_point_pix, self.schema_points[14][1]]
        self.schema_points = np.vstack([self.schema_points, np_21])

        left_bask = [
            ((self.schema_points[2][0] + self.schema_points[4][0]) / 2) + new_bask_pix,
            self.schema_points[2][1]
            + ((self.schema_points[4][1] - self.schema_points[2][1]) / 2),
        ]
        self.schema_points = np.vstack([self.schema_points, left_bask])

        right_bask = [
            ((self.schema_points[10][0] + self.schema_points[12][0]) / 2)
            - new_bask_pix,
            self.schema_points[10][1]
            + ((self.schema_points[12][1] - self.schema_points[10][1]) / 2),
        ]
        self.schema_points = np.vstack([self.schema_points, right_bask])

        # Sidelines
        # TOP
        draw_line_between_points(
            warped_img, self.schema_points[0], self.schema_points[16]
        )
        draw_line_between_points(
            warped_img, self.schema_points[16], self.schema_points[8]
        )
        # RIGHT
        draw_line_between_points(
            warped_img, self.schema_points[8], self.schema_points[10]
        )
        draw_line_between_points(
            warped_img, self.schema_points[10], self.schema_points[12]
        )
        draw_line_between_points(
            warped_img, self.schema_points[12], self.schema_points[15]
        )
        # BOTTOM
        draw_line_between_points(
            warped_img, self.schema_points[15], self.schema_points[17]
        )
        draw_line_between_points(
            warped_img, self.schema_points[17], self.schema_points[7]
        )
        # LEFT
        draw_line_between_points(
            warped_img, self.schema_points[7], self.schema_points[4]
        )
        draw_line_between_points(
            warped_img, self.schema_points[4], self.schema_points[2]
        )
        draw_line_between_points(
            warped_img, self.schema_points[2], self.schema_points[0]
        )
        # MID LANE
        draw_line_between_points(
            warped_img, self.schema_points[16], self.schema_points[17]
        )

        # Lane lines
        # LEFT
        draw_line_between_points(
            warped_img, self.schema_points[2], self.schema_points[3]
        )
        draw_line_between_points(
            warped_img, self.schema_points[3], self.schema_points[5]
        )
        draw_line_between_points(
            warped_img, self.schema_points[5], self.schema_points[4]
        )
        # RIGHT
        draw_line_between_points(
            warped_img, self.schema_points[10], self.schema_points[11]
        )
        draw_line_between_points(
            warped_img, self.schema_points[11], self.schema_points[13]
        )
        draw_line_between_points(
            warped_img, self.schema_points[13], self.schema_points[12]
        )

        # 3-pts lines
        draw_line_between_points(
            warped_img, self.schema_points[1], self.schema_points[18]
        )
        draw_line_between_points(
            warped_img, self.schema_points[6], self.schema_points[19]
        )
        draw_line_between_points(
            warped_img, self.schema_points[9], self.schema_points[20]
        )
        draw_line_between_points(
            warped_img, self.schema_points[14], self.schema_points[21]
        )

        radius_left = int(
            np.linalg.norm(
                np.array(self.schema_points[18]) - np.array(self.schema_points[19])
            )
            / 2
        )
        radius_left = (
            radius_left + 5
        )  # Need explanations for the small offset incrementation

        radius_right = int(
            np.linalg.norm(
                np.array(self.schema_points[20]) - np.array(self.schema_points[21])
            )
            / 2
        )
        radius_right = (
            radius_right + 5
        )  # Need explanations for the small offset incrementation

        # Half Circle
        center_left = tuple(map(int, self.schema_points[22]))
        cv2.ellipse(
            warped_img,
            center_left,
            (radius_left, radius_left),
            -100,
            20,
            180,
            (0, 255, 0),
            1,
        )

        center_right = tuple(map(int, self.schema_points[23]))
        cv2.ellipse(
            warped_img,
            center_right,
            (radius_right, radius_right),
            100,
            0,
            160,
            (0, 255, 0),
            1,
        )

        return warped_img

    # def generate_tracking_points(self):
    #     """
    #     DEPRECATED:
    #     Generate tracking_points equally spaced on each detected line
    #     """
    #     # Sideline segments
    #     self.tracking_points.append(generate_points_on_line(self.schema_points[0], self.schema_points[1], 20))
    #     self.tracking_points.append(generate_points_on_line(self.schema_points[1], self.schema_points[2], 20))
    #     self.tracking_points.append(generate_points_on_line(self.schema_points[2], self.schema_points[3], 20))
    #     self.tracking_points.append(generate_points_on_line(self.schema_points[3], self.schema_points[0], 20))
    #
    #     # 3-pt lines
    #     self.tracking_points.append(generate_points_on_line(self.schema_points[4], self.schema_points[5], 10))
    #     self.tracking_points.append(generate_points_on_line(self.schema_points[6], self.schema_points[7], 10))
    #
    #     # Lane lines
    #     self.tracking_points.append(generate_points_on_line(self.schema_points[8], self.schema_points[9], 20))
    #     self.tracking_points.append(generate_points_on_line(self.schema_points[10], self.schema_points[11], 20))
    #     self.tracking_points.append(generate_points_on_line(self.schema_points[11], self.schema_points[8], 20))
    #
    #     # Flatten to a single array
    #     self.tracking_points = np.vstack(self.tracking_points).astype(np.float32)
