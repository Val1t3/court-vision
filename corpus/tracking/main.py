from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Tuple

import cv2
from ultralytics import YOLO


def analyze_video_with_yolov11_bytetrack(
    video_path: str,
    output_video_path: Optional[str] = None,
    output_csv_path: Optional[str] = None,
    *,
    model_weights: str = "../data/models/yolov11m.pt",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> Tuple[str, str]:
    """Detect and track *persons* in a video using YOLOv11m + ByteTrack.

    Parameters
    ----------
    video_path: str
        Path to the input video file.
    output_video_path: Optional[str]
        Where to save the annotated/tracked video (MP4). If omitted, a file named
        "<input_name>_tracked.mp4" will be created next to the input.
    output_csv_path: Optional[str]
        Where to save the CSV with per-frame bounding boxes. If omitted, a file named
        "<input_name>_tracks.csv" will be created next to the input.
    model_weights: str
        YOLO weights to use. Defaults to the COCO-pretrained YOLOv11m weights.
        The file will be downloaded automatically by Ultralytics on first use.
    conf_threshold: float
        Detector confidence threshold.
    iou_threshold: float
        IoU threshold for NMS.

    Returns
    -------
    Tuple[str, str]
        (annotated_video_path, csv_path)

    Notes
    -----
    - Tracking uses ByteTrack via Ultralytics' built-in tracker configuration.
    - Only the COCO "person" class (index 0) is tracked and exported.
    - The CSV columns are: frame, track_id, x1, y1, x2, y2, confidence.
    """

    source_path = Path(video_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Input video not found: {source_path}")

    annotated_video_path = (
        Path(output_video_path) if output_video_path else source_path.with_name(f"{source_path.stem}_tracked.mp4")
    )
    csv_path = Path(output_csv_path) if output_csv_path else source_path.with_name(f"{source_path.stem}_tracks.csv")

    annotated_video_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Load YOLOv11 model (downloads weights on first run, if needed)
    model = YOLO(model_weights)

    # Probe video for dimensions and FPS
    probe = cv2.VideoCapture(str(source_path))
    if not probe.isOpened():
        raise RuntimeError(f"Failed to open video: {source_path}")
    fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    probe.release()

    # Video writer (MP4)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # widely compatible; use "avc1" if you have H.264
    video_writer = cv2.VideoWriter(str(annotated_video_path), fourcc, float(fps), (frame_width, frame_height))

    # CSV writer
    csv_file = open(csv_path, mode="w", newline="")
    csv_columns = ["frame", "track_id", "x1", "y1", "x2", "y2", "confidence"]
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
    csv_writer.writeheader()

    frame_index = -1

    try:
        # Stream results frame-by-frame, running ByteTrack. Filter to "person" class only.
        results_generator = model.track(
            source=str(source_path),
            tracker="bytetrack.yaml",
            conf=conf_threshold,
            iou=iou_threshold,
            classes=[0],  # COCO class 0 == person
            stream=True,
            persist=True,
            verbose=False,
        )

        for result in results_generator:
            frame_index += 1

            # Draw boxes + IDs on the frame and write to the output video
            annotated_frame = result.plot()  # returns a BGR numpy array
            if annotated_frame is None:
                # Fallback to the original frame if plotting failed for any reason
                annotated_frame = result.orig_img
            if annotated_frame is not None:
                video_writer.write(annotated_frame)

            # Extract tracked boxes with IDs
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            # When tracking, boxes.id holds the tracker IDs. It can be None for a few initial frames.
            track_ids = getattr(boxes, "id", None)
            if track_ids is None:
                continue

            # Safely move tensors to CPU lists if necessary
            to_list = lambda t: t.cpu().tolist() if hasattr(t, "cpu") else (t.tolist() if hasattr(t, "tolist") else t)

            ids_list = to_list(track_ids)
            xyxy_list = to_list(boxes.xyxy)
            conf_list = to_list(boxes.conf)

            for tid, (x1, y1, x2, y2), conf in zip(ids_list, xyxy_list, conf_list):
                csv_writer.writerow(
                    {
                        "frame": frame_index,
                        "track_id": int(tid) if tid is not None else -1,
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                        "confidence": float(conf),
                    }
                )

    finally:
        video_writer.release()
        csv_file.close()

    return str(annotated_video_path), str(csv_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect and track persons in a video using YOLOv11m + ByteTrack, saving an annotated MP4 and a CSV of bounding boxes."
    )
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("--out-video", type=str, default=None, help="Path to save the annotated video (MP4).")
    parser.add_argument("--out-csv", type=str, default=None, help="Path to save the tracks CSV.")
    parser.add_argument("--weights", type=str, default="../data/models/yolo11m.pt", help="YOLO weights to use.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detection.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS.")

    args = parser.parse_args()

    video_out, csv_out = analyze_video_with_yolov11_bytetrack(
        args.video_path,
        output_video_path=args.out_video,
        output_csv_path=args.out_csv,
        model_weights=args.weights,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )

    print(f"Annotated video saved to: {video_out}")
    print(f"Tracks CSV saved to: {csv_out}")
