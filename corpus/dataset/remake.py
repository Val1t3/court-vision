# file: person_sahi_tracker.py
from __future__ import annotations

import os
import cv2
import math
import time
import uuid
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

# SAHI + Ultralytics
from sahi.predict import get_sliced_prediction
from sahi.auto_model import AutoDetectionModel


# ------------------------------- Utility types -------------------------------

@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    category_name: str

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    score: float
    last_seen_frame_index: int
    time_since_update: int = 0
    hits: int = 1
    age: int = 0


# ------------------------------- Geometry helpers ----------------------------

def compute_iou_matrix(tracks: List[Track], detections: List[Detection]) -> np.ndarray:
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)), dtype=np.float32)

    track_boxes = np.array([t.bbox for t in tracks], dtype=np.float32)  # N x 4
    det_boxes = np.array([d.to_xyxy() for d in detections], dtype=np.float32)  # M x 4

    # Compute IoU between each track and detection
    ious = np.zeros((track_boxes.shape[0], det_boxes.shape[0]), dtype=np.float32)

    for i, tb in enumerate(track_boxes):
        tx1, ty1, tx2, ty2 = tb
        t_area = max(0.0, tx2 - tx1) * max(0.0, ty2 - ty1)
        for j, db in enumerate(det_boxes):
            dx1, dy1, dx2, dy2 = db
            d_area = max(0.0, dx2 - dx1) * max(0.0, dy2 - dy1)
            ix1 = max(tx1, dx1)
            iy1 = max(ty1, dy1)
            ix2 = min(tx2, dx2)
            iy2 = min(ty2, dy2)
            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            inter = iw * ih
            union = t_area + d_area - inter + 1e-6
            ious[i, j] = inter / union
    return ious


# ------------------------------- Simple tracker ------------------------------

class IOUHungarianTracker:
    """
    Minimal, dependency-light tracker:
      - State = latest bbox (no Kalman prediction)
      - Association = Hungarian on 1 - IoU cost
      - New tracks created for unmatched detections
      - Old tracks removed if not matched for 'max_age' frames

    This is intentionally simple and robust for many CCTV-like videos.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 1,
        starting_track_id: int = 1,
    ):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.next_track_id = starting_track_id
        self.tracks: List[Track] = []

    def update(self, detections: List[Detection], frame_index: int) -> List[Track]:
        # Age and mark time since update
        for t in self.tracks:
            t.age += 1
            t.time_since_update += 1

        if len(detections) == 0:
            # Remove stale tracks
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return self.tracks

        # Associate with Hungarian
        iou_matrix = compute_iou_matrix(self.tracks, detections)
        if iou_matrix.size > 0:
            cost = 1.0 - iou_matrix
            row_idx, col_idx = linear_sum_assignment(cost)
        else:
            row_idx, col_idx = np.array([], dtype=int), np.array([], dtype=int)

        # Determine matches above IoU threshold
        matched_indices = []
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_detections = list(range(len(detections)))

        for r, c in zip(row_idx, col_idx):
            if iou_matrix[r, c] >= self.iou_threshold:
                matched_indices.append((r, c))
        matched_track_indices = set([m[0] for m in matched_indices])
        matched_det_indices = set([m[1] for m in matched_indices])

        unmatched_tracks = [i for i in unmatched_tracks if i not in matched_track_indices]
        unmatched_detections = [i for i in unmatched_detections if i not in matched_det_indices]

        # Update matched tracks
        for t_idx, d_idx in matched_indices:
            det = detections[d_idx]
            self.tracks[t_idx].bbox = np.array(det.to_xyxy(), dtype=np.float32)
            self.tracks[t_idx].score = det.score
            self.tracks[t_idx].last_seen_frame_index = frame_index
            self.tracks[t_idx].time_since_update = 0
            self.tracks[t_idx].hits += 1

        # Create new tracks for unmatched detections
        for d_idx in unmatched_detections:
            det = detections[d_idx]
            new_track = Track(
                track_id=self.next_track_id,
                bbox=np.array(det.to_xyxy(), dtype=np.float32),
                score=det.score,
                last_seen_frame_index=frame_index,
                time_since_update=0,
                hits=1,
                age=0,
            )
            self.next_track_id += 1
            self.tracks.append(new_track)

        # Remove stale tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Only return tracks that have at least min_hits (helps reduce ID flicker at start)
        visible_tracks = [t for t in self.tracks if t.hits >= self.min_hits]
        return visible_tracks


# ------------------------------- SAHI + YOLOv11m -----------------------------

def build_sahi_ultralytics_model(
    model_path: str = "yolo11m.pt",
    device: str = "cuda:0" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu",
    confidence_threshold: float = 0.25,
) -> AutoDetectionModel:
    """
    Creates a SAHI AutoDetectionModel wrapping an Ultralytics YOLOv11m model.
    """
    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        device=device,
    )


def run_sahi_detection_on_frame(
    bgr_frame: np.ndarray,
    detection_model: AutoDetectionModel,
    slice_height: int = 640,
    slice_width: int = 640,
    overlap_ratio: float = 0.2,
    category_name_filter: Optional[List[str]] = None,
) -> List[Detection]:
    """
    Runs SAHI slicing detection on a single BGR frame.
    Filters to specified category names if provided.
    Returns a list of Detection objects in original image coordinates.
    """
    # SAHI expects RGB images
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

    result = get_sliced_prediction(
        image=rgb,
        detection_model=detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        perform_standard_pred=False,  # slice-only since we specifically want SAHI behavior
        verbose=0,
    )

    detections: List[Detection] = []
    for obj in result.object_prediction_list:
        cat_name = obj.category.name if obj.category is not None else ""
        if category_name_filter and cat_name not in category_name_filter:
            continue
        # bbox in VOC [xmin, ymin, xmax, ymax]
        x1, y1, x2, y2 = obj.bbox.to_voc_bbox()
        score = float(obj.score.value) if obj.score is not None else 0.0
        detections.append(Detection(x1, y1, x2, y2, score, cat_name))
    return detections


# ------------------------------- Main pipeline -------------------------------

def analyze_video_with_yolo11m_sahi(
    video_path: str,
    output_video_path: Optional[str] = None,
    output_csv_path: Optional[str] = None,
    model_path: str = "../data/models/yolo11m.pt",
    device: str = "cuda:0" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu",
    conf_threshold: float = 0.25,
    sahi_slice_height: int = 1200,
    sahi_slice_width: int = 1200,
    sahi_overlap_ratio: float = 0.2,
    tracker_iou_threshold: float = 0.3,
    tracker_max_age: int = 30,
    tracker_min_hits: int = 1,
    draw_thickness: int = 2,
    draw_font_scale: float = 0.6,
) -> Tuple[str, str]:
    """
    Analyze a video and return paths to the annotated video and CSV with per-frame person tracks.

    Parameters
    ----------
    video_path: str
        Path to the input video file.
    output_video_path: Optional[str]
        Where to write the annotated video. If None, will be alongside the input file with suffix "_analyzed.mp4".
    output_csv_path: Optional[str]
        Where to write the CSV. If None, will be alongside the input file with suffix "_tracks.csv".
    model_path: str
        Ultralytics YOLOv11m weights (default 'yolo11m.pt').
    device: str
        'cuda:0' if you have GPU, else 'cpu'.
    conf_threshold: float
        Confidence threshold for detection.
    sahi_slice_height/width: int
        SAHI tile size.
    sahi_overlap_ratio: float
        Overlap fraction between tiles (0..1).
    tracker_*:
        Parameters for the internal IoU + Hungarian tracker.

    Returns
    -------
    (output_video_path, output_csv_path)
    """

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    stem, _ = os.path.splitext(video_path)
    if output_video_path is None:
        output_video_path = f"{stem}_analyzed.mp4"
    if output_csv_path is None:
        output_csv_path = f"{stem}_tracks.csv"

    # Build detection model (SAHI wrapping Ultralytics YOLOv11m)
    detection_model = build_sahi_ultralytics_model(
        model_path=model_path,
        device=device,
        confidence_threshold=conf_threshold,
    )

    # Only keep "person" category
    person_category = ["person"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Video writer (MP4V widely supported)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps if fps > 0 else 25.0, (width, height))

    tracker = IOUHungarianTracker(
        iou_threshold=tracker_iou_threshold,
        max_age=tracker_max_age,
        min_hits=tracker_min_hits,
    )

    csv_rows: List[Dict] = []
    rng = tqdm(total=total_frames if total_frames > 0 else None, desc="Processing", unit="frame")

    frame_index = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1) Detect persons with SAHI slicing + YOLOv11m
            detections = run_sahi_detection_on_frame(
                frame,
                detection_model,
                slice_height=sahi_slice_height,
                slice_width=sahi_slice_width,
                overlap_ratio=sahi_overlap_ratio,
                category_name_filter=person_category,
            )

            # 2) Track across frames
            tracks = tracker.update(detections, frame_index)

            # 3) Draw + write rows
            for t in tracks:
                x1, y1, x2, y2 = [int(v) for v in t.bbox]
                # Clamp to frame
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))

                color = (37, 255, 127)  # green-ish
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, draw_thickness)
                label = f"id {t.track_id}"
                cv2.putText(
                    frame,
                    label,
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    draw_font_scale,
                    color,
                    max(1, draw_thickness - 1),
                    cv2.LINE_AA,
                )

                # write CSV row
                csv_rows.append(
                    {
                        "frame_index": frame_index,
                        "track_id": t.track_id,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "score": float(t.score),
                    }
                )

            out.write(frame)
            frame_index += 1
            rng.update(1)
    finally:
        rng.close()
        cap.release()
        out.release()

    # Dump CSV
    df = pd.DataFrame(csv_rows, columns=["frame_index", "track_id", "x1", "y1", "x2", "y2", "score"])
    df.to_csv(output_csv_path, index=False)

    return output_video_path, output_csv_path


# ------------------------------- Script usage --------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Person detection+tracking with YOLOv11m + SAHI.")
    parser.add_argument("video_path", type=str, help="Path to input video")
    parser.add_argument("--model", type=str, default="yolo11m.pt", help="Ultralytics YOLOv11m weights")
    parser.add_argument("--device", type=str, default=None, help="cuda:0 or cpu (auto if omitted)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--slice_h", type=int, default=1200, help="SAHI slice height")
    parser.add_argument("--slice_w", type=int, default=1200, help="SAHI slice width")
    parser.add_argument("--overlap", type=float, default=0.2, help="SAHI overlap ratio (0..1)")
    parser.add_argument("--out_video", type=str, default=None, help="Output video path")
    parser.add_argument("--out_csv", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    dev = args.device if args.device is not None else ("cuda:0" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu")
    analyzed_video, csv_path = analyze_video_with_yolo11m_sahi(
        video_path=args.video_path,
        output_video_path=args.out_video,
        output_csv_path=args.out_csv,
        model_path=args.model,
        device=dev,
        conf_threshold=args.conf,
        sahi_slice_height=args.slice_h,
        sahi_slice_width=args.slice_w,
        sahi_overlap_ratio=args.overlap,
    )
    print("Annotated video:", analyzed_video)
    print("Tracks CSV:", csv_path)
