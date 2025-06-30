# court-vision/src, main.py
# Code written by Valentin Woehrel, 2025

from baseline_detection import BaselineDetection
from ultralytics.engine.results import Results
import csv
from typing import List
import numpy as np
import cv2
from player_detection_sahi import player_detection_sahi


# constants
model = "models/yolo11m.pt"
source = "assets/extract-3.mp4"
player_positions_path = "saves/player_positions.csv"
point_positions_path = "saves/point_positions.csv"
output_path = "output/test.mp4"


def save_positions(results: List[Results]) -> None:
    """
    Export results in a .csv file, write position of every detected players at
    each frame.

    Parameters
    ----------
    results : List[Results]
        List of `result` (detected boxes) at each frame.
    """
    with open(player_positions_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame", "id", "x1", "y1", "x2", "y2", "confidence", "class"])

        frame_idx = 0
        for result in results:
            boxes = result.boxes
            if boxes.id is None:
                continue  # no tracking info

            for i in range(len(boxes)):
                track_id = int(boxes.id[i].item())
                cls = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()

                writer.writerow([frame_idx, track_id, x1, y1, x2, y2, conf, cls])

            frame_idx += 1


def convert_to_schema_env(
        coord_path : str,
        output_path : str,
        homography : np.ndarray
    ) -> None:
    """
    Convert boxes coordinates to points coordinates with an application of the
    homography matrix. Write coordinates in a new .csv file.

    Parameters
    ----------
    coord_path : str
        Path to the player coordinates file.
    output_path : str
        Path to store the new coordinates.
    homography : np.ndarray
        Homography matrix to use.
    """
    # read player_positions
    with open(coord_path, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
        rows_new = []

        # create new adapted rows for point coordinates
        for row in rows:
            # calculate the bottom-center point of the bounding box
            x = float(row['x1']) + (float(row['x2']) - float(row['x1'])) / 2
            y = float(row['y2'])
            point = np.array([[x, y]], dtype=np.float32)

            # apply homography
            point_hom = cv2.perspectiveTransform(point[None, :, :], homography)[0, 0]
            x_h, y_h = point_hom[0], point_hom[1]

            # add new row
            rows_new.append({
                'frame': row['frame'],
                'id': row['id'],
                'x': x_h,
                'y': y_h,
                'confidence': row['confidence'],
                'class': row['class'],
            })

        # save rows in a new .csv file
        with open(output_path, "w", newline="") as outfile:
            fieldnames = ['frame', 'id', 'x', 'y', 'confidence', 'class']
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows_new:
                writer.writerow(row)


def draw_schematic_position(
        point_positions_path : str,
        output_path : str
    ) -> None:
    # Read point positions from CSV
    points_by_frame = {}
    with open(point_positions_path, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            frame = int(row['frame'])
            x = float(row['x'])
            y = float(row['y'])
            pid = int(row['id'])
            cls = int(row['class'])
            conf = float(row['confidence'])
            if frame not in points_by_frame:
                points_by_frame[frame] = []
            points_by_frame[frame].append({
                'x': x, 'y': y, 'id': pid, 'class': cls, 'confidence': conf
            })

    # get total number of frames
    max_frame = max(points_by_frame.keys())
    # load the background image
    bg_img = cv2.imread("assets/cropped_schema.png")
    if bg_img is None:
        raise FileNotFoundError("Background image not found at assets/cropped_schema.png")
    h, w, _ = bg_img.shape

    # set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))

    for frame_idx in range(max_frame + 1):
        frame_img = bg_img.copy()
        points = points_by_frame.get(frame_idx, [])
        for pt in points:
            center = (int(pt['x']), int(pt['y']))
            color = (0, 255, 0) if pt['class'] == 0 else (0, 0, 255)
            cv2.circle(frame_img, center, 8, color, -1)
            cv2.putText(frame_img, str(pt['id']), (center[0]+10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        out.write(frame_img)

    out.release()


if __name__ == "__main__":
    # set-up BaselineDetection class, and calculate homography between
    # first_frame and schema.
    bd = BaselineDetection(
        frame_points_path="data/frame_points_fix.json",
        schema_points_path="data/points_cropped_schema.json"
    )
    h, h_inv = bd.calculate_homography()


    ### OLD VERSION WITH YOLO TRACKING ###
    # # compute player detection, and export coords in a .csv file
    # model = YOLO(model)

    # # run tracking
    # print("run tracking...")
    # results = model.track(
    #     source=source,
    #     persist=True,
    #     classes=0,  # person class
    #     tracker="bytetrack.yaml",
    #     save=True
    # )

    # print("save positions...")
    # save_positions(results=results)
    ######################################

    print("detect players...")
    player_detection_sahi(
        video=source,
        model=model,
        points=[
            [476, 457],
            [1442, 470],
            [129, 670],
            [1779, 686]
        ],
        video_output="output/main_extract_3.mp4",
        results_path=player_positions_path,
    )

    print("convert positions to schema env...")
    convert_to_schema_env(
        coord_path=player_positions_path,
        output_path=point_positions_path,
        homography=h
    )

    print("draw positions on schema...")
    draw_schematic_position(
        point_positions_path=point_positions_path,
        output_path=output_path
    )

    print("exit 0")