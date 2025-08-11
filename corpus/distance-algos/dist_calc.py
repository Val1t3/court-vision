"""
Player Distance with Homography — Euclidean, Kalman, Optical Flow (meters)

Reads:
- Detections CSV with bounding boxes per frame (frame,id,x1,y1,x2,y2,...)
- Two JSON files containing *the same* N court landmarks:
    • image_points.json  — pixel coords in the **video frame**
    • court_points.json  — coords of the same landmarks, either in **meters** or **schema pixels** from a court diagram

Builds a homography H : image(px) → court(m). Projects each detection's **feet point** to court meters
and computes per‑player distances using:
  1) Euclidean (frame‑to‑frame on projected points)
  2) Kalman‑smoothed (constant‑velocity KF in meters)
  3) Optical Flow (optional): tracks in image but accumulates **meters** by projecting each step

CLI example (your uploaded files):

python player_distance_homography.py \
  --csv /path/to/detections.csv \
  --img-points /mnt/data/frame_points_fix.json \
  --court-points /mnt/data/points_cropped_schema.json \
  --court-units schema_px \
  --schema-length-m 28.0 \
  --schema-width-m 15.0 \
  --video /path/to/video.mp4   # optional; needed for optical flow
  --output distances.csv

JSON formats accepted for the two correspondence files:
- [ [x,y], [x,y], ... ]
- [ {"x": x, "y": y}, ... ]
- {"points": [ ... one of the above ... ]}

Notes
- If your court JSON is already in **meters**, use --court-units meters and skip length/width flags.
- If it's from a **schema image** (like your 797×429 diagram), use --court-units schema_px and provide real court size.
- Use ≥4 well‑distributed pairs; 8–18 across the court is ideal.
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json

import numpy as np
import pandas as pd

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


# -----------------------------
# IO and parsing
# -----------------------------


def load_detections(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"frame", "id", "x1", "y1", "x2", "y2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    df = df.copy()
    df["frame"] = df["frame"].astype(int)
    df["id"] = df["id"].astype(int)
    if "confidence" not in df.columns:
        df["confidence"] = 1.0
    return df


def _coerce_points(obj) -> np.ndarray:
    if isinstance(obj, dict) and "points" in obj:
        obj = obj["points"]
    pts: List[Tuple[float, float]] = []
    for item in obj:
        if isinstance(item, dict) and "x" in item and "y" in item:
            pts.append((float(item["x"]), float(item["y"])))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            pts.append((float(item[0]), float(item[1])))
        else:
            raise ValueError(
                "Unsupported point format in JSON; expected [x,y] or {x,y}."
            )
    return np.array(pts, dtype=np.float32)


def load_points_json(path: str) -> np.ndarray:
    with open(path, "r") as f:
        data = json.load(f)
    pts = _coerce_points(data)
    if pts.shape[0] < 4:
        raise ValueError(
            "Need at least 4 correspondence points to compute a homography."
        )
    return pts


# -----------------------------
# Geometry and projection
# -----------------------------


def compute_homography(
    image_points_xy: np.ndarray, court_points_xy_m: np.ndarray
) -> np.ndarray:
    if image_points_xy.shape != court_points_xy_m.shape:
        raise ValueError("Image and court point arrays must have the same shape.")
    if image_points_xy.shape[0] < 4:
        raise ValueError("At least 4 point pairs required for homography.")
    H, mask = cv2.findHomography(
        image_points_xy.astype(np.float32),
        court_points_xy_m.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
    )
    if H is None:
        raise RuntimeError("cv2.findHomography failed. Check your correspondences.")
    return H


def warp_points_to_court(points_xy_px: np.ndarray, H: np.ndarray) -> np.ndarray:
    if points_xy_px.ndim == 1:
        points_xy_px = points_xy_px.reshape(1, 2)
    ones = np.ones((points_xy_px.shape[0], 1), dtype=np.float64)
    pts_h = np.concatenate([points_xy_px.astype(np.float64), ones], axis=1).T  # (3,N)
    warped = H.astype(np.float64) @ pts_h  # (3,N)
    warped /= warped[2:3, :]
    return warped[:2, :].T.astype(np.float64)  # (N,2) meters


def bbox_feet_point(row: pd.Series, mode: str = "bottom_center") -> Tuple[float, float]:
    x1, y1, x2, y2 = (
        float(row["x1"]),
        float(row["y1"]),
        float(row["x2"]),
        float(row["y2"]),
    )
    if mode in {"bottom_center", "bottom_mid", "feet"}:
        cx = 0.5 * (x1 + x2)
        by = y2
        return cx, by
    elif mode == "center":
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        return cx, cy
    else:
        raise ValueError(f"Unsupported feet-point mode: {mode}")


# -----------------------------
# Distances on projected (meter) tracks with detailed output
# -----------------------------


def distance_euclidean_meters_detailed(df_proj: pd.DataFrame) -> pd.DataFrame:
    """Returns DataFrame with (frame, id, step_m, cum_m, method)"""
    results = []

    for pid, g in df_proj.sort_values(["id", "frame"]).groupby("id"):
        g = g.reset_index(drop=True)
        arr = g[["X", "Y"]].to_numpy(dtype=np.float64)
        frames = g["frame"].to_numpy()

        if len(arr) < 2:
            # Single frame case
            results.append(
                {
                    "frame": int(frames[0]),
                    "id": int(pid),
                    "step_m": 0.0,
                    "cum_m": 0.0,
                    "method": "euclidean",
                }
            )
            continue

        # Calculate step distances
        deltas = np.sqrt(np.sum(np.diff(arr, axis=0) ** 2, axis=1))
        cum_dist = 0.0

        # First frame
        results.append(
            {
                "frame": int(frames[0]),
                "id": int(pid),
                "step_m": 0.0,
                "cum_m": 0.0,
                "method": "euclidean",
            }
        )

        # Subsequent frames
        for i, delta in enumerate(deltas):
            cum_dist += float(delta)
            results.append(
                {
                    "frame": int(frames[i + 1]),
                    "id": int(pid),
                    "step_m": float(delta),
                    "cum_m": cum_dist,
                    "method": "euclidean",
                }
            )

    return pd.DataFrame(results)


@dataclass
class KFConfig:
    fps: float = 25.0
    process_var: float = 5.0
    meas_var: float = 0.04  # meters^2 (after projection); tune to your jitter level
    conf_meas_floor: float = 0.1


def _kf_mats(dt: float, q_mag: float):
    A = np.array(
        [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
    )
    dt2, dt3, dt4 = dt * dt, dt * dt * dt, dt * dt * dt * dt
    q = q_mag
    Q = (
        np.array(
            [
                [dt4 / 4, 0, dt3 / 2, 0],
                [0, dt4 / 4, 0, dt3 / 2],
                [dt3 / 2, 0, dt2, 0],
                [0, dt3 / 2, 0, dt2],
            ],
            dtype=float,
        )
        * q
    )
    return A, Q


def _meas_noise_from_conf(conf: float, base_var: float, conf_floor: float) -> float:
    c = max(conf_floor, min(1.0, float(conf)))
    return base_var / c


def distance_kalman_meters_detailed(
    df_proj: pd.DataFrame, cfg: KFConfig
) -> pd.DataFrame:
    """Returns DataFrame with (frame, id, step_m, cum_m, method)"""
    Hm = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    I = np.eye(4)
    dt = 1.0 / max(1e-6, cfg.fps)
    A, Q = _kf_mats(dt, cfg.process_var)

    results = []

    for pid, g in df_proj.sort_values(["id", "frame"]).groupby("id"):
        g = g.reset_index(drop=True)
        arr = g[["X", "Y"]].to_numpy(dtype=float)
        frames = g["frame"].to_numpy()

        if len(arr) < 2:
            # Single frame case
            results.append(
                {
                    "frame": int(frames[0]),
                    "id": int(pid),
                    "step_m": 0.0,
                    "cum_m": 0.0,
                    "method": "kalman",
                }
            )
            continue

        confs = (
            g["confidence"].to_numpy(dtype=float)
            if "confidence" in g.columns
            else np.ones(len(arr))
        )

        x = np.array([arr[0, 0], arr[0, 1], 0.0, 0.0], dtype=float)
        P = np.eye(4) * 10.0
        smoothed: List[Tuple[float, float]] = [(arr[0, 0], arr[0, 1])]

        # First frame
        results.append(
            {
                "frame": int(frames[0]),
                "id": int(pid),
                "step_m": 0.0,
                "cum_m": 0.0,
                "method": "kalman",
            }
        )

        cum_dist = 0.0
        for i in range(1, len(arr)):
            gap = int(frames[i] - frames[i - 1])
            gap = max(1, gap)
            for _ in range(gap):
                x = A @ x
                P = A @ P @ A.T + Q
            z = arr[i]
            R_var = _meas_noise_from_conf(confs[i], cfg.meas_var, cfg.conf_meas_floor)
            R = np.eye(2) * R_var
            y = z - (Hm @ x)
            S = Hm @ P @ Hm.T + R
            K = P @ Hm.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (I - K @ Hm) @ P

            current_pos = (float(x[0]), float(x[1]))
            smoothed.append(current_pos)

            # Calculate step distance from previous smoothed position
            prev_pos = smoothed[-2]
            step_dist = float(
                np.hypot(current_pos[0] - prev_pos[0], current_pos[1] - prev_pos[1])
            )
            cum_dist += step_dist

            results.append(
                {
                    "frame": int(frames[i]),
                    "id": int(pid),
                    "step_m": step_dist,
                    "cum_m": cum_dist,
                    "method": "kalman",
                }
            )

    return pd.DataFrame(results)


# -----------------------------
# Optical Flow accumulation in meters with detailed output
# -----------------------------


@dataclass
class OFConfig:
    win_size: Tuple[int, int] = (21, 21)
    max_level: int = 3
    criteria: Tuple[int, int, float] = (3 | 1, 30, 0.01)
    reinit_on_detection: bool = True
    detection_blend: float = 0.3


def distance_optical_flow_meters_detailed(
    video_path: str,
    df_det: pd.DataFrame,
    H: np.ndarray,
    of_cfg: OFConfig = OFConfig(),
) -> pd.DataFrame:
    """Returns DataFrame with (frame, id, step_m, cum_m, method)"""
    if cv2 is None:
        raise ImportError(
            "OpenCV (cv2) is required for optical flow but not available."
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    frames_group = {k: v for k, v in df_det.groupby("frame")}

    pts_img: Dict[int, np.ndarray] = {}
    valid: Dict[int, bool] = {}
    cumulative_m: Dict[int, float] = {}
    results = []

    # Initialize cumulative distances for all players
    for pid in df_det["id"].unique():
        cumulative_m[pid] = 0.0

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        raise IOError("Failed to read first video frame.")
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_idx = 0

    # Initialize first frame
    if frame_idx in frames_group:
        for _, row in frames_group[frame_idx].iterrows():
            pid = int(row["id"])
            cx, by = row["feet_x"], row["feet_y"]
            pts_img[pid] = np.array([[[cx, by]]], dtype=np.float32)
            valid[pid] = True
            results.append(
                {
                    "frame": frame_idx,
                    "id": pid,
                    "step_m": 0.0,
                    "cum_m": 0.0,
                    "method": "optical_flow",
                }
            )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_idx += 1

        if not pts_img:
            if frame_idx in frames_group:
                for _, row in frames_group[frame_idx].iterrows():
                    pid = int(row["id"])
                    cx, by = row["feet_x"], row["feet_y"]
                    pts_img[pid] = np.array([[[cx, by]]], dtype=np.float32)
                    valid[pid] = True
                    results.append(
                        {
                            "frame": frame_idx,
                            "id": pid,
                            "step_m": 0.0,
                            "cum_m": cumulative_m[pid],
                            "method": "optical_flow",
                        }
                    )
            prev_gray = gray
            continue

        pids = [pid for pid, ok in valid.items() if ok]
        if pids:
            p0 = np.vstack([pts_img[pid] for pid in pids])
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                gray,
                p0,
                None,
                winSize=of_cfg.win_size,
                maxLevel=of_cfg.max_level,
                criteria=of_cfg.criteria,
            )
            for j, pid in enumerate(pids):
                step_dist = 0.0
                if st[j] == 1:
                    old_px = p0[j, 0]
                    new_px = p1[j, 0]
                    old_xy_m = warp_points_to_court(old_px, H)[0]
                    new_xy_m = warp_points_to_court(new_px, H)[0]
                    step_dist = float(np.hypot(*(new_xy_m - old_xy_m)))
                    cumulative_m[pid] += step_dist

                    if frame_idx in frames_group:
                        g = frames_group[frame_idx]
                        det = g[g["id"] == pid]
                        if len(det) > 0:
                            cx, by = (
                                float(det.iloc[0]["feet_x"]),
                                float(det.iloc[0]["feet_y"]),
                            )
                            if of_cfg.reinit_on_detection:
                                pts_img[pid] = np.array([[[cx, by]]], dtype=np.float32)
                            else:
                                blend = of_cfg.detection_blend
                                pts_img[pid] = np.array(
                                    [
                                        [
                                            [
                                                (1 - blend) * new_px[0] + blend * cx,
                                                (1 - blend) * new_px[1] + blend * by,
                                            ]
                                        ]
                                    ],
                                    dtype=np.float32,
                                )
                        else:
                            pts_img[pid] = np.array(
                                [[[new_px[0], new_px[1]]]], dtype=np.float32
                            )
                    else:
                        pts_img[pid] = np.array(
                            [[[new_px[0], new_px[1]]]], dtype=np.float32
                        )
                    valid[pid] = True
                else:
                    valid[pid] = False

                results.append(
                    {
                        "frame": frame_idx,
                        "id": pid,
                        "step_m": step_dist,
                        "cum_m": cumulative_m[pid],
                        "method": "optical_flow",
                    }
                )

        # Handle new detections
        if frame_idx in frames_group:
            g = frames_group[frame_idx]
            for _, row in g.iterrows():
                pid = int(row["id"])
                if pid not in valid or not valid[pid]:
                    pts_img[pid] = np.array(
                        [[[row["feet_x"], row["feet_y"]]]], dtype=np.float32
                    )
                    valid[pid] = True
                    # Add entry if not already added above
                    if not any(
                        r["frame"] == frame_idx and r["id"] == pid for r in results
                    ):
                        results.append(
                            {
                                "frame": frame_idx,
                                "id": pid,
                                "step_m": 0.0,
                                "cum_m": cumulative_m[pid],
                                "method": "optical_flow",
                            }
                        )

        prev_gray = gray

    cap.release()
    return pd.DataFrame(results)


# Keep original functions for backward compatibility
def distance_euclidean_meters(df_proj: pd.DataFrame) -> Dict[int, float]:
    df_detailed = distance_euclidean_meters_detailed(df_proj)
    return df_detailed.groupby("id")["cum_m"].max().to_dict()


def distance_kalman_meters(df_proj: pd.DataFrame, cfg: KFConfig) -> Dict[int, float]:
    df_detailed = distance_kalman_meters_detailed(df_proj, cfg)
    return df_detailed.groupby("id")["cum_m"].max().to_dict()


def distance_optical_flow_meters(
    video_path: str,
    df_det: pd.DataFrame,
    H: np.ndarray,
    of_cfg: OFConfig = OFConfig(),
) -> Dict[int, float]:
    df_detailed = distance_optical_flow_meters_detailed(video_path, df_det, H, of_cfg)
    return df_detailed.groupby("id")["cum_m"].max().to_dict()


# -----------------------------
# Helpers for units & metrics
# -----------------------------


def convert_schema_px_to_meters(
    court_pts_schema: np.ndarray, length_m: float, width_m: float
) -> np.ndarray:
    """Map court diagram pixels to meters using axis-aligned scaling from extents.
    Assumes x spans court length and y spans court width on the schema image.
    """
    xs = court_pts_schema[:, 0]
    ys = court_pts_schema[:, 1]
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    length_px = max(1e-6, x_max - x_min)
    width_px = max(1e-6, y_max - y_min)
    sx = length_m / length_px
    sy = width_m / width_px
    pts_m = np.empty_like(court_pts_schema, dtype=np.float32)
    pts_m[:, 0] = (court_pts_schema[:, 0] - x_min) * sx
    pts_m[:, 1] = (court_pts_schema[:, 1] - y_min) * sy
    return pts_m.astype(np.float32)


def reprojection_rmse(H: np.ndarray, img_pts: np.ndarray, tgt_pts: np.ndarray) -> float:
    pred = warp_points_to_court(img_pts, H)
    err = pred - tgt_pts
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))


# -----------------------------
# Orchestration
# -----------------------------


def prepare_projected_dataframe(
    detections: pd.DataFrame,
    H: np.ndarray,
    feet_mode: str = "bottom_center",
) -> pd.DataFrame:
    df = detections.copy()
    feet_xy = df.apply(
        lambda r: bbox_feet_point(r, feet_mode), axis=1, result_type="expand"
    )
    df["feet_x"], df["feet_y"] = feet_xy[0].astype(float), feet_xy[1].astype(float)
    XYm = warp_points_to_court(df[["feet_x", "feet_y"]].to_numpy(dtype=np.float64), H)
    df["X"], df["Y"] = XYm[:, 0], XYm[:, 1]
    return df[["frame", "id", "X", "Y", "confidence", "feet_x", "feet_y"]]


def run_all(
    csv_path: str,
    img_points_path: str,
    court_points_path: str,
    fps: float = 25.0,
    video_path: Optional[str] = None,
    feet_point: str = "bottom_center",
    court_units: str = "schema_px",
    schema_length_m: float = 28.0,
    schema_width_m: float = 15.0,
    output_csv: Optional[str] = None,
) -> Dict[str, Dict[int, float]]:
    det = load_detections(csv_path)
    img_pts = load_points_json(img_points_path)
    court_pts_in = load_points_json(court_points_path)

    if court_units == "schema_px":
        court_pts_m = convert_schema_px_to_meters(
            court_pts_in, schema_length_m, schema_width_m
        )
    elif court_units == "meters":
        court_pts_m = court_pts_in.astype(np.float32)
    else:
        raise ValueError("court_units must be 'meters' or 'schema_px'")

    H = compute_homography(img_pts, court_pts_m)

    # Calibration quality
    rmse = reprojection_rmse(H, img_pts, court_pts_m)
    print(f"Homography reprojection RMSE: {rmse:.3f} m over {len(img_pts)} points")

    df_proj = prepare_projected_dataframe(det, H, feet_point)

    # Get detailed results for CSV output
    eucl_detailed = distance_euclidean_meters_detailed(df_proj)
    kf_detailed = distance_kalman_meters_detailed(df_proj, KFConfig(fps=fps))

    # Combine all detailed results
    all_detailed = [eucl_detailed, kf_detailed]

    if video_path:
        of_detailed = distance_optical_flow_meters_detailed(video_path, df_proj, H)
        all_detailed.append(of_detailed)

    # Save detailed CSV if output path provided
    if output_csv:
        combined_df = pd.concat(all_detailed, ignore_index=True)
        combined_df = combined_df.sort_values(["method", "id", "frame"])
        combined_df.to_csv(output_csv, index=False)
        print(f"Detailed distance data saved to: {output_csv}")

    # Return summary results for backward compatibility
    eucl = eucl_detailed.groupby("id")["cum_m"].max().to_dict()
    kf = kf_detailed.groupby("id")["cum_m"].max().to_dict()

    results: Dict[str, Dict[int, float]] = {
        "euclidean_m": eucl,
        "kalman_m": kf,
    }

    if video_path:
        of = of_detailed.groupby("id")["cum_m"].max().to_dict()
        results["optical_flow_m"] = of

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Player distance in meters using homography (Euclidean, Kalman, Optical Flow)"
    )
    parser.add_argument("--csv", required=True, help="Path to detections CSV")
    parser.add_argument(
        "--img-points", required=True, help="JSON with image points (pixels)"
    )
    parser.add_argument(
        "--court-points",
        required=True,
        help="JSON with corresponding court points (meters or schema pixels)",
    )
    parser.add_argument(
        "--video", default=None, help="Path to video (required for optical flow)"
    )
    parser.add_argument(
        "--fps", type=float, default=25.0, help="FPS (used if --video omitted)"
    )
    parser.add_argument(
        "--feet-point",
        choices=["bottom_center", "bottom_mid", "feet", "center"],
        default="bottom_center",
    )
    parser.add_argument(
        "--court-units",
        choices=["meters", "schema_px"],
        default="schema_px",
        help="Units used in court points JSON. If schema_px, provide physical dimensions.",
    )
    parser.add_argument(
        "--schema-length-m",
        type=float,
        default=28.0,
        help="Physical court length in meters (x axis) when court-units=schema_px",
    )
    parser.add_argument(
        "--schema-width-m",
        type=float,
        default=15.0,
        help="Physical court width in meters (y axis) when court-units=schema_px",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output CSV file path for detailed distance data (frame,id,step_m,cum_m,method)",
    )

    args = parser.parse_args()

    # If video provided, try reading real FPS for better Kalman dt
    def get_fps_from_video(vpath: Optional[str], fallback: float) -> float:
        if not vpath:
            return fallback
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            return fallback
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return float(fps) if fps and fps > 0 else fallback

    fps = get_fps_from_video(args.video, args.fps)

    results = run_all(
        csv_path=args.csv,
        img_points_path=args.img_points,
        court_points_path=args.court_points,
        fps=fps,
        video_path=args.video,
        feet_point=args.feet_point,
        court_units=args.court_units,
        schema_length_m=args.schema_length_m,
        schema_width_m=args.schema_width_m,
        output_csv=args.output,
    )

    def fmt_meters(block: Dict[int, float]) -> str:
        lines = []
        for pid in sorted(block.keys()):
            lines.append(f"  id {pid:>3}: {block[pid]:8.2f} m")
        return "".join(lines) if lines else "  (no players)"

    print("Euclidean distance (meters):" + fmt_meters(results["euclidean_m"]))
    print("Kalman-smoothed distance (meters):" + fmt_meters(results["kalman_m"]))

    if "optical_flow_m" in results:
        print("Optical flow distance (meters):" + fmt_meters(results["optical_flow_m"]))
    else:
        print("Optical flow skipped (no --video provided).")


if __name__ == "__main__":
    main()
