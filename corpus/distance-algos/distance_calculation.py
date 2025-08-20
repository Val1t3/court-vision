from euclidean import Euclidean
from kalman import Kalman
from optical_flow import OpticalFlow
import cv2
import numpy as np
from baseline_detection import BaselineDetection


# consts
name = "eval_2"
video_src = "../data/assets/" + name + ".mov"
csv_path = "../data/courtvision-dataset/" + name + "_tracks.csv"
frame_points_path = "../data/data/eval_points.json"
schema_points_path = "../data/data/points_cropped_schema.json"


def get_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


if __name__ == "__main__":
    bd = BaselineDetection(
        frame_points_path=frame_points_path,
        schema_points_path=schema_points_path,
    )

    h, h_inv = bd.calculate_homography()

    Euclidean(
        csv_path=csv_path,
        schema_points_path=schema_points_path,
        h=h,
    )

    # Kalman(
    #     csv_path=csv_path, schema_points_path=schema_points_path, fps=get_fps(video_src)
    # )
    # OpticalFlow(
    #     csv_path="../data/saves/player_positions_eval_8.csv",
    #     video_path="../data/assets/eval_8.mov",
    #     schema_points_path=schema_points_path,
    # )
