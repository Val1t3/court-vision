# court-vision/src, player_detection_sahi.py
# Code written by Valentin Woehrel, 2025

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import visualize_object_predictions
import cv2
import argparse
import json
import numpy as np


# TODO:
# [X] Retrieve result object list
# [X] Filter with person category only
# [X] Draw boxes and texts on the image
# [ ] Check if box in the area of the court


# constants
version = "11n"

video = "assets/extract-4.mp4"
model = "models/yolo" + version +".pt"
output = "output/player-detection-" + version + ".mp4"
show = True

points = []
# 0: top-left
# 7: bottom-left
# 8: top-right
# 15: bottom-right


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="Process input file")
    parser.add_argument('--points', type=str, required=False, help="Coordinates" \
    "of court points")

    args = parser.parse_args()

    # if points file given, open it and store coord in `points` variable
    if args.points:
        try:
            with open(args.points, 'r') as f:
                points = np.array(json.load(f), dtype=np.float64).tolist()
        except:
            print("error: Impossible to open points file.")
            exit(1)

    # create model detection
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model,
        confidence_threshold=0.3,
        device="cpu",
    )

    # open the input video
    cap = cv2.VideoCapture(video)

    # Video writer to save the output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output, fourcc, fps, (width, height))

    # loop through frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # predict using sliced prediction method
        result = get_sliced_prediction(
            frame,
            model,
            slice_height=800,
            slice_width=800,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            verbose=2,
        )

        # filter predictions to keep only "person" category
        result.object_prediction_list = [
            item for item in result.object_prediction_list
            if item.category.name == "person"
        ]

        # filter keeping boxes in court ares only
        if args.points:
            # margin = ((points[12][1] - points[10][1])
            #           + (points[4][1] - points[2][1])) / 2
            margin = 0

            # filter boxes not in the court area + margin
            filtered_pred = []
            for i in result.object_prediction_list:
                # convert bbox into 4 points
                bbox = i.bbox
                minx, miny, maxx, maxy = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy

                if (miny < max(points[7][1], points[15][1])
                    and maxy > min(points[0][1], points[8][0])
                    and minx < max(points[8][0], points[15][0])
                    and maxx > min(points[0][0], points[7][0])):
                    filtered_pred.append(i)

            result.object_prediction_list = filtered_pred

        # draw on image detected persons
        res = visualize_object_predictions(
            image=frame,
            object_prediction_list=result.object_prediction_list,
            rect_th=2,
            text_size=1,
            text_th=1
        )

        # show frame (optional)
        if show is True:
            cv2.imshow("SAHI Player Detection", res["image"])

        # write the frame to the output video
        out.write(res["image"])

        # press 'q' to exit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
