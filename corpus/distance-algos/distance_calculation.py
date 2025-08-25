from euclidean import Euclidean
from kalman import KalmanDistance
from optical_flow import OpticalFlowDistanceSimple, OpticalFlowDistanceAdvanced
import cv2
from baseline_detection import BaselineDetection
import argparse


def get_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


if __name__ == "__main__":
    # arg parser
    parser = argparse.ArgumentParser(description="Distance calculation for court vision dataset")
    parser.add_argument("--name", type=str, help="Name of the evaluation set")

    args = parser.parse_args()

    name = args.name

    # const
    video_src = "../data/assets/" + name + ".mov"
    csv_path = "../data/courtvision-dataset/" + name + "_tracks.csv"
    frame_points_path = "../data/data/eval_points.json"
    schema_points_path = "../data/data/points_cropped_schema.json"

    # script
    bd = BaselineDetection(
        frame_points_path=frame_points_path,
        schema_points_path=schema_points_path,
    )

    h, h_inv = bd.calculate_homography()

    Euclidean(
        csv_path=csv_path,
        schema_points_path=schema_points_path,
        h=h,
        output_csv_path=csv_path.replace("_tracks.csv", "_dist_euclidean.csv")
    )

    KalmanDistance(
        csv_path=csv_path,
        schema_points_path=schema_points_path,
        h=h,
        process_noise=1.0,
        measurement_noise=10.0,
        output_csv_path=csv_path.replace("_tracks.csv", "_dist_kalman.csv")
    )

    OpticalFlowDistanceSimple(
        csv_path=csv_path,
        video_path=video_src,
        schema_points_path=schema_points_path,
        h=h,
        output_csv_path=csv_path.replace("_tracks.csv", "_dist_optical_flow.csv")
    )
