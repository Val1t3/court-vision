# ~/Code/court-vision/src, player_tracking_bytetrack.py
# Code written by Valentin Woehrel, 2025

from ultralytics import YOLO
import csv


# constants
model = "models/yolo11m.pt"
source = "assets/extract-4.mp4"
csv_path = "saves/player_position.csv"


if __name__ == "__main__":
    # load the YOLOv11 model
    model = YOLO(model)

    # run tracking
    results = model.track(
        source=source,
        persist=True,
        classes=0,  # person class
        tracker="bytetrack.yaml",
        save=True,
    )

    # save position of person on each frame
    with open(csv_path, "w", newline="") as csvfile:
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
