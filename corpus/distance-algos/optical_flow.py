import numpy as np
import pandas as pd
import cv2
from scale import Scale
from baseline_detection import apply_homography

class OpticalFlowDistanceSimple:
    """
    Simple optical flow distance calculation using dense optical flow.

    Tracks player movement using Farneback optical flow method.
    """

    def __init__(self, csv_path: str, video_path: str, schema_points_path: str, 
                 h: np.ndarray, output_csv_path: str, roi_size: int = 30):
        scale = Scale(schema_points_path)

        self.df = pd.read_csv(csv_path)
        self.df = self.df.sort_values(by=["frame"])
        self.df1 = self.df[self.df['id'] == 1]

        self.output_path = output_csv_path
        self.video_path = video_path
        self.h = h
        self.scale = scale.scale
        self.roi_size = roi_size

        distance1 = self._calculate_simple_optical_flow()

        print(f"Optical Flow distance: {distance1:.2f} meters")


    def _export_results(self, frames: list, step_distances: list, cum_distances: list,
                    output_csv_path: str):
        results = pd.DataFrame({
            'frame': frames,
            'id': 1,
            'step_m': step_distances,
            'cum_m': cum_distances,
            'method': 'opical_flow'
        })

        results.to_csv(output_csv_path, index=False)


    def _calculate_simple_optical_flow(self):
        """Calculate distance using simple dense optical flow."""
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_path}")

        total_distance = 0.0
        step_distances = [0.0]
        cum_distances = [0.0]
        frames = [0]
        prev_gray = None
        prev_center = None

        for _, row in self.df1.iterrows():
            frame_num = int(row['frame'])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()

            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Get player center
            center_x = int(row['x1'] + (row['x2'] - row['x1']) / 2)
            center_y = int(row['y2'])
            current_center = np.array([center_x, center_y])

            if prev_gray is not None and prev_center is not None:
                # Calculate dense optical flow
                flow = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray,
                    prev_center.reshape(1, 1, 2).astype(np.float32),
                    None
                )[0]

                if flow is not None:
                    # Get the tracked position
                    tracked_pos = flow[0][0]

                    # Apply homography to both positions
                    prev_transformed = apply_homography(
                        pt=tuple(prev_center), h_matrix=self.h
                    )
                    current_transformed = apply_homography(
                        pt=tuple(tracked_pos), h_matrix=self.h
                    )

                    # Calculate distance
                    distance = np.sqrt(
                        (current_transformed[0] - prev_transformed[0]) ** 2 +
                        (current_transformed[1] - prev_transformed[1]) ** 2
                    ) * self.scale

                    total_distance += distance
                    frames.append(row['frame'])
                    step_distances.append(distance)
                    cum_distances.append(total_distance)

            prev_gray = gray.copy()
            prev_center = current_center

        cap.release()
        self._export_results(
            frames=frames,
            step_distances=step_distances,
            cum_distances=cum_distances,
            output_csv_path=self.output_path
        )
        print(f"Results written to {self.output_path}")

        return total_distance


class OpticalFlowDistanceAdvanced:
    """
    Advanced optical flow distance calculation with feature detection.

    Uses corner detection around player position for more robust tracking.
    """

    def __init__(self, csv_path: str, video_path: str, schema_points_path: str,
                 h: np.ndarray, roi_size: int = 50):
        scale = Scale(schema_points_path)

        self.df = pd.read_csv(csv_path)
        self.df = self.df.sort_values(by=["frame"])
        self.df1 = self.df[self.df['id'] == 1]

        self.video_path = video_path
        self.h = h
        self.scale = scale.scale
        self.roi_size = roi_size

        distance1 = self._calculate_feature_based_distance()

        print(f"Total Feature-based Optical Flow distance for player 1: {distance1:.2f} meters")

        self.distance1 = distance1

    def _calculate_feature_based_distance(self):
        """Calculate distance using feature-based optical flow."""
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_path}")

        total_distance = 0.0
        prev_gray = None
        prev_features = None

        # Parameters for corner detection
        feature_params = dict(
            maxCorners=10,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )

        # Parameters for Lucas-Kanade optical flow
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        for _, row in self.df1.iterrows():
            frame_num = int(row['frame'])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()

            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Get player center
            center_x = int(row['x1'] + (row['x2'] - row['x1']) / 2)
            center_y = int(row['y2'])

            # Define ROI around player
            roi_x1 = max(0, center_x - self.roi_size)
            roi_y1 = max(0, center_y - self.roi_size)
            roi_x2 = min(gray.shape[1], center_x + self.roi_size)
            roi_y2 = min(gray.shape[0], center_y + self.roi_size)

            if prev_gray is not None and prev_features is not None:
                # Track features using optical flow
                new_features, status, error = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, prev_features, None, **lk_params
                )

                # Keep only good features
                good_new = new_features[status == 1]
                good_old = prev_features[status == 1]

                if len(good_new) > 0:
                    # Calculate average displacement
                    displacement = np.mean(good_new - good_old, axis=0)

                    # Apply displacement to previous center
                    new_center = np.array([center_x, center_y]) + displacement
                    prev_center = np.array([center_x, center_y])

                    # Apply homography
                    prev_transformed = apply_homography(
                        pt=tuple(prev_center), h_matrix=self.h
                    )
                    current_transformed = apply_homography(
                        pt=tuple(new_center), h_matrix=self.h
                    )

                    # Calculate distance
                    distance = np.sqrt(
                        (current_transformed[0] - prev_transformed[0]) ** 2 +
                        (current_transformed[1] - prev_transformed[1]) ** 2
                    ) * self.scale

                    total_distance += distance

            # Detect new features in ROI
            roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
            features = cv2.goodFeaturesToTrack(roi, mask=None, **feature_params)

            if features is not None:
                # Convert ROI coordinates back to full image coordinates
                features[:, :, 0] += roi_x1
                features[:, :, 1] += roi_y1
                prev_features = features
            else:
                # Fallback to center point
                prev_features = np.array([[[center_x, center_y]]], dtype=np.float32)

            prev_gray = gray.copy()

        cap.release()
        return total_distance