# court-vision/src, court_tracker.py
# Code written by Valentin Woehrel, 2025

import numpy as np
import cv2


class CourtTracker:
    """
    A class used to track important points on the court.

    Attributes
    ----------
    schema_pts : np.array
    current_side : str
    """

    def __init__(self, schema_points, side: str):
        self.schema_pts = np.array(schema_points, dtype=np.float32)
        self.left_schema_pts = self._get_left_side_pts()
        self.right_schema_pts = self._get_right_side_pts()

        self.current_side = side
        self.anchor_frame_pts = None
        self.court_simulation = None

    def _get_left_side_pts(self):
        return np.array(
            [
                self.schema_pts[0],  # Sideline top-left
                self.schema_pts[1],  # Sideline top-left mid
                self.schema_pts[2],  # Lane top-left (near free throw)
                self.schema_pts[3],  # Lane bottom-left (near baseline)
                self.schema_pts[4],  # Sideline bottom-left
                self.schema_pts[5],  # Sideline bottom-left mide
            ],
            dtype=np.float32,
        ).reshape(-1, 1, 2)

    def _get_right_side_pts(self):
        return np.array(
            [
                self.schema_pts[6],  # Sideline top-right
                self.schema_pts[7],  # Sideline top-right mid
                self.schema_pts[8],  # Lane top-right (near free throw)
                self.schema_pts[9],  # Lane bottom-right (near baseline)
                self.schema_pts[10],  # Sideline bottom-right
                self.schema_pts[11],  # Sideline bottom-right mide
            ],
            dtype=np.float32,
        ).reshape(-1, 1, 2)

    def simulate_pts(self, points: np.ndarray, h_inv: np.ndarray) -> np.ndarray:
        """
        Simulates missing points based on the inverse homography

        pts : array of size of self.schema_pts, and pts who needs to be simulated hahe coordinates 10000, 10000

        Returns
        -------
            points: np.ndarray of shape (-1, 2), with missing points filled.
        """

        # Define indexes
        mask_null = (points[:, 0] == 10000.0) & (points[:, 1] == 10000.0)
        indexes = np.where(mask_null)[0]

        if len(indexes) == 0:
            return points

        # Select schema points
        schema_pts = np.array([self.schema_pts[i] for i in indexes])
        schema_pts = schema_pts.reshape(-1, 1, 2)

        # Convert schema_pts to homogeneous coordinates
        sim_frame_pts = cv2.perspectiveTransform(schema_pts, h_inv)
        sim_frame_pts = sim_frame_pts.reshape(-1, 2)

        # Replace null values with simulated
        for i, sim in enumerate(sim_frame_pts):
            points[indexes[i]] = sim

        return points

    def detect_visible_side(self, frame_shape: tuple, h_inv: tuple):
        """
        Estimate which court side is currently cisible in the frame.
        """

        h, w = frame_shape[:2]
        left_pts_in_frame = cv2.perspectiveTransform(self.left_schema_pts, h_inv)
        right_pts_in_frame = cv2.perspectiveTransform(self.right_schema_pts, h_inv)

        def points_in_frame(pts):
            return np.all(
                (pts[:, 0, 0] >= 0)
                & (pts[:, 0, 0] < w)
                & (pts[:, 0, 1] >= 0)
                & (pts[:, 0, 1] < h)
            )

        if points_in_frame(left_pts_in_frame):
            self.current_side = "left"
            self.anchor_frame_pts = self.left_schema_pts
            self.anchor_frame_pts = left_pts_in_frame
        elif points_in_frame(right_pts_in_frame):
            self.current_side = "right"
            self.anchor_frame_pts = self.right_schema_pts
            self.anchor_frame_pts = right_pts_in_frame
        else:
            self.current_side = None
            self.anchor_frame_pts = None
            self.anchor_frame_pts = None

        return self.current_side

    def track(self, prev_gray, curr_gray):
        if self.anchor_frame_pts is None:
            return None, None

        lk_params = dict(
            winSize=(21, 21),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        next_pts, st, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, self.anchor_frame_pts, None, **lk_params
        )

        good_prev = self.anchor_frame_pts[st == 1].reshape(-1, 2)
        good_next = next_pts[st == 1].reshape(-1, 2)

        # TRACKING FILTER
        displacements = []

        for i, (prev_pt, next_pt) in enumerate(zip(good_prev, good_next)):
            displacements.append(np.linalg.norm(next_pt - prev_pt))

        median_disp = np.median(displacements)  # Compute median of displacements
        mad_disp = np.median(
            np.abs(displacements - median_disp)
        )  # Compute median absolute deviation
        treshold = median_disp + 3 * mad_disp  # MODIFY THE CONSTANT FOR PRECISION

        for i, disp in enumerate(displacements):
            if disp > treshold:
                good_next[i] = good_prev[i]
                print("#######################################")
                print(
                    "MODIF:", good_next[i], "DISPLACEMENT:", disp, "TRESHOLD:", treshold
                )

        if len(good_next) >= 4:
            H, _ = cv2.findHomography(
                self.anchor_frame_pts[st == 1],
                good_next.reshape(-1, 1, 2),
                cv2.RANSAC,
                5.0,
            )
            self.anchor_frame_pts = good_next.reshape(-1, 1, 2)  # Update for next frame
            return H, good_next
        else:
            return None, None

    def calculate_homography_side(self, frame_points, schema_points, side: str):
        # Side parameter error handling
        if side not in ["left", "right"]:
            raise ValueError(
                "[calculate_homography_side error]: wrong value for side parameter"
            )

        # Frame_points & schema_points parameters error handling
        if len(frame_points) < 4 or len(schema_points) < 4:
            raise ValueError(
                "[calculate_homography_side error]: not enough points to calculate homography"
            )

        # Select schema points corresponding to the chosen side
        if side == "left":
            schema_indexes = [0, 1, 2, 3, 4, 5]
        else:
            schema_indexes = [6, 7, 8, 9, 10, 11]

        selected_schema_points = np.array(
            [schema_points[i] for i in schema_indexes], dtype=np.float32
        )

        # Calculate homography matrix
        h, _ = cv2.findHomography(frame_points, selected_schema_points)
        h_inv, _ = cv2.findHomography(selected_schema_points, frame_points)

        if h is None or h_inv is None:
            raise ValueError(
                "[calculate_homography_side error]: couldn't calculate homography matrix"
            )

        return h, h_inv
