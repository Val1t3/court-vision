import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
from ultralytics import YOLO


@dataclass
class TrackRecord:
    frame_index: int
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


def analyze_video_with_yolo8_bytetrack(
    video_path: str,
    output_video_path: Optional[str] = None,
    output_csv_path: Optional[str] = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    device: Optional[str] = None,  # e.g., "0" for GPU, or "cpu"
    project_dir: Optional[
        str
    ] = None,  # where Ultralytics will write the annotated video
    tracker_yaml: str = "bytetrack.yaml",  # use Ultralytics' built-in ByteTrack config
) -> Tuple[str, str]:
    """
    Detects and tracks persons in a video using YOLOv8m + ByteTrack.
    Returns (annotated_video_path, csv_path).

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    output_video_path : Optional[str]
        Desired path for the annotated video. If None, the Ultralytics default path is used and returned.
    output_csv_path : Optional[str]
        Desired path for the CSV. If None, it will be created next to the output video.
    conf_threshold : float
        Detection confidence threshold.
    iou_threshold : float
        NMS IoU threshold.
    device : Optional[str]
        "cpu" or CUDA device string like "0". If None, Ultralytics will auto-select.
    project_dir : Optional[str]
        Directory for Ultralytics run artifacts (default is `runs/track`).
    tracker_yaml : str
        Tracker config. Keep "bytetrack.yaml" to use ByteTrack.
    """

    # --- Validate inputs
    input_path = Path(video_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    # --- Load model (auto-downloads weights if needed)
    # Model name must match Ultralytics' YOLOv8 medium checkpoint
    model = YOLO("yolo8m.pt")

    # We want only person class (COCO class 0). Using classes=[0] filters detections to "person".
    # `stream=True` yields per-frame results. `save=True` writes annotated video to disk.
    # `persist=True` keeps the tracker across frames.
    stream = model.track(
        source=str(input_path),
        tracker=tracker_yaml,
        conf=conf_threshold,
        iou=iou_threshold,
        classes=[0],  # person only
        stream=True,
        save=True,
        device=device,
        project=project_dir,  # None -> default "runs/track"
        name=None,  # Ultralytics will choose exp/expN name automatically unless provided
        verbose=False,
        persist=True,
    )

    # We will:
    # 1) iterate the generator
    # 2) collect track rows
    # 3) capture the save directory from the first result
    track_rows: List[TrackRecord] = []
    save_dir: Optional[Path] = None
    out_video_auto_path: Optional[Path] = None
    frame_index = -1

    for results in stream:
        frame_index += 1

        # Resolve save_dir once
        if save_dir is None:
            # Most Ultralytics results expose `save_dir` (Path) and `path` (input frame source)
            try:
                save_dir = Path(results.save_dir)
            except Exception:
                # Fallback: use default runs directory
                save_dir = Path("runs") / "track"
            # Try to infer the video file Ultralytics is writing
            # Typically it's save_dir / input_video_name_with_ext
            out_video_auto_path = _guess_output_video_path(save_dir, input_path.name)

        # Each `results` corresponds to a frame; extract tracked boxes
        if results.boxes is None or len(results.boxes) == 0:
            continue

        boxes = results.boxes
        # `boxes.id` holds track IDs when using a tracker; may be None for untracked detections
        track_ids = boxes.id
        confidences = boxes.conf
        xyxy = boxes.xyxy

        if track_ids is None:
            # If tracker hasn't initialized yet, skip this frame
            continue

        for i in range(len(xyxy)):
            track_id_tensor = track_ids[i]
            if track_id_tensor is None:
                continue

            # Tensors -> python types
            track_id = int(track_id_tensor.item())
            x1, y1, x2, y2 = [float(v.item()) for v in xyxy[i]]
            conf = float(confidences[i].item()) if confidences is not None else 0.0

            track_rows.append(
                TrackRecord(
                    frame_index=frame_index,
                    track_id=track_id,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=conf,
                )
            )

    if save_dir is None:
        # No frames processed
        raise RuntimeError("No frames were processed; check the input video or codecs.")

    # Decide final video path
    if output_video_path:
        final_video_path = Path(output_video_path)
        if out_video_auto_path and out_video_auto_path.exists():
            final_video_path.parent.mkdir(parents=True, exist_ok=True)
            # Move/rename the file to the requested location
            out_video_auto_path.replace(final_video_path)
        else:
            # If we couldn't guess or find it, try to locate any video in save_dir
            guessed = _find_first_video_file(save_dir)
            if guessed is None:
                raise RuntimeError(
                    f"Ultralytics did not produce an annotated video under {save_dir}."
                )
            final_video_path.parent.mkdir(parents=True, exist_ok=True)
            Path(guessed).replace(final_video_path)
    else:
        # Keep the auto-saved path if available; otherwise pick the first video in save_dir
        if out_video_auto_path and out_video_auto_path.exists():
            final_video_path = out_video_auto_path
        else:
            guessed = _find_first_video_file(save_dir)
            if guessed is None:
                raise RuntimeError(
                    f"Ultralytics did not produce an annotated video under {save_dir}."
                )
            final_video_path = Path(guessed)

    # Write CSV
    if output_csv_path:
        csv_path = Path(output_csv_path)
    else:
        csv_path = final_video_path.with_suffix("").with_name(
            final_video_path.stem + "_tracks.csv"
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    _write_tracks_csv(csv_path, track_rows)

    return str(final_video_path), str(csv_path)


def _write_tracks_csv(csv_path: Path, rows: List[TrackRecord]) -> None:
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "id", "x1", "y1", "x2", "y2", "confidence"])
        for r in rows:
            writer.writerow(
                [r.frame_index, r.track_id, r.x1, r.y1, r.x2, r.y2, r.confidence]
            )


def _guess_output_video_path(save_dir: Path, input_name: str) -> Optional[Path]:
    """
    Ultralytics typically saves the annotated video as save_dir / <input_name_with_ext>.
    This function checks that convention first, then falls back to scanning the directory.
    """
    candidate = save_dir / input_name
    if candidate.exists() and candidate.is_file():
        return candidate
    # Some setups may re-encode to .avi or .mp4 regardless of source extension; try common variants
    stem = Path(input_name).stem
    for ext in [".mp4", ".avi", ".mov", ".mkv"]:
        alt = save_dir / f"{stem}{ext}"
        if alt.exists():
            return alt
    # Fallback: scan for the first video file
    return _find_first_video_file(save_dir)


def _find_first_video_file(directory: Path) -> Optional[Path]:
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
    for path in sorted(directory.glob("*")):
        if path.suffix.lower() in video_exts and path.is_file():
            return path
    # Recurse one level (Ultralytics sometimes nests exp dirs)
    for sub in sorted(directory.glob("*")):
        if sub.is_dir():
            found = _find_first_video_file(sub)
            if found:
                return found
    return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python yolo8_bytetrack_person_tracker.py <video_path> "
            "[out_video_path] [out_csv_path]"
        )
        sys.exit(1)

    inp = sys.argv[1]
    out_vid = sys.argv[2] if len(sys.argv) > 2 else None
    out_csv = sys.argv[3] if len(sys.argv) > 3 else None

    annotated_video, csv_file = analyze_video_with_yolo8_bytetrack(
        video_path=inp,
        output_video_path=out_vid,
        output_csv_path=out_csv,
        conf_threshold=0.25,
        iou_threshold=0.5,
        device=None,  # change to "0" to force first CUDA GPU, or "cpu"
        project_dir=None,  # or set a custom directory for runs
        tracker_yaml="bytetrack.yaml",
    )

    print("Annotated video:", annotated_video)
    print("CSV:", csv_file)
