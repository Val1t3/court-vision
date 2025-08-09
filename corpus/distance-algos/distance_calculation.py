from euclidean import Euclidean
from kalman import Kalman
from optical_flow import OpticalFlow


# consts
name = "eval_8"
csv_path = "../data/saves/smoothed_positions_" + name + ".csv"
schema_points_path = "../data/data/points_cropped_schema.json"


if __name__ == "__main__":
    Euclidean(
        csv_path=csv_path,
        schema_points_path=schema_points_path
    )
    Kalman(
        csv_path=csv_path,
        schema_points_path=schema_points_path
    )
    OpticalFlow(
        csv_path="../data/saves/player_positions_eval_8.csv",
        video_path="../data/assets/eval_8.mov",
        schema_points_path=schema_points_path
    )
