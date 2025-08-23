import cv2
import numpy as np
import pandas as pd
from baseline_detection import BaselineDetection


# retrieve FPS from a given video file
def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def create_video(name: str):
    coordinates_path = '../data/courtvision-dataset/' + name + '_tracks.csv'
    output_video = '../data/output/schema_position_' + name + '.mp4'
    input_video_path = '../data/courtvision-dataset/' + name + '.mov'
    schema_path = '../data/assets/cropped_schema.png'

    schema_img = cv2.imread(schema_path)
    frame_size = (schema_img.shape[1], schema_img.shape[0])  # Width, Height

    bd = BaselineDetection(
        frame_points_path="../data/data/eval_points.json",
        schema_points_path="../data/data/points_cropped_schema.json"
    )
    h, _ = bd.calculate_homography()
    fps = get_video_fps(input_video_path)
    df = pd.read_csv(coordinates_path)
    df = df[df['id'] == 1].sort_values('frame')

    # create list of coordinates
    coordinates = []
    for item in df.itertuples():
        x = item.x1 + (item.x2 - item.x1) / 2
        y = item.y2
        coordinates.append(bd.apply_homography((x, y), h))

    # video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

    # draw each frame
    for pos in coordinates:
        frame = schema_img.copy()
        x, y = int(pos[0]), int(pos[1])
        cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)  # Draw player position
        video_writer.write(frame)

    video_writer.release()


if __name__ == "__main__":
    create_video('eval_1')
    print(f'Video saved into ../data/output folder')
