"""
Basketball Player Movement Distance — Euclidean vs Kalman vs Optical Flow

This script loads frame-by-frame detections from a CSV and computes per-player
travel distances using three approaches:

1) Euclidean (frame-to-frame center distance)
2) Kalman-smoothed trajectory (constant-velocity KF, then distance on smoothed path)
3) Optical Flow (Lucas–Kanade, using video frames). Requires the matching video.

CSV format (one detection per row):
frame,id,x1,y1,x2,y2,confidence,class

Where (x1,y1) is top-left, (x2,y2) is bottom-right of a player bounding box.

Outputs: dictionaries keyed by player id with distances in pixels (and optionally meters).

Usage (CLI):

python test.py \
  --csv /path/to/detections.csv \
  --video /path/to/video.mp4 \
  --fps 25 \
  --pixels-per-meter 18.0

If --video is omitted, optical flow is skipped.

Dependencies: pandas, numpy, opencv-python

"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from scale import Scale

import numpy as np
import pandas as pd

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


# -----------------------------
# Utils
# -----------------------------

def load_detections(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"frame", "id", "x1", "y1", "x2", "y2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    # Ensure types
    df = df.copy()
    df["frame"] = df["frame"].astype(int)
    df["id"] = df["id"].astype(int)
    # Compute centers
    df["cx"] = (df["x1"] + df["x2"]) / 2.0
    df["cy"] = (df["y1"] + df["y2"]) / 2.0
    if "confidence" not in df.columns:
        df["confidence"] = 1.0
    return df


def to_meters(distance_px: float, m_per_px: Optional[float]) -> float:
    return distance_px * m_per_px if m_per_px and m_per_px > 0 else distance_px

# -----------------------------
# 1) Euclidean distance
# -----------------------------

def distance_euclidean(df: pd.DataFrame, m_per_px: Optional[float] = None) -> Dict[int, Dict[str, float]]:
    """Compute per-player total path length by summing frame-to-frame center deltas.

    Returns {id: {"pixels": total_px, "meters": total_m}}.
    """
    out: Dict[int, Dict[str, float]] = {}
    for pid, g in df.sort_values(["id", "frame"]).groupby("id"):
        g = g[["frame", "cx", "cy"]].reset_index(drop=True)
        # Only consecutive frames; if gaps exist, we still measure between the last seen and next seen positions
        deltas = np.sqrt(np.diff(g["cx"].to_numpy()) ** 2 + np.diff(g["cy"].to_numpy()) ** 2)
        total_px = float(np.nansum(deltas))
        out[pid] = {
            "pixels": total_px,
            "meters": float(to_meters(total_px, m_per_px)),
        }
    return out


# -----------------------------
# 2) Kalman filter smoothing (constant-velocity)
# -----------------------------

@dataclass
class KFConfig:
    fps: float = 25.0
    process_var: float = 50.0  # process noise magnitude (tune)
    meas_var: float = 25.0     # default measurement noise variance (pixels^2)
    conf_meas_floor: float = 0.1  # minimum confidence considered


def _kf_mats(dt: float, q_mag: float) -> Tuple[np.ndarray, np.ndarray]:
    """State x=[x,y,vx,vy]. Return (A, Q) for constant-velocity model."""
    A = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=float)
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt
    q = q_mag
    Q = np.array([
        [dt4/4, 0,     dt3/2, 0    ],
        [0,     dt4/4, 0,     dt3/2],
        [dt3/2, 0,     dt2,   0    ],
        [0,     dt3/2, 0,     dt2  ],
    ], dtype=float) * q
    return A, Q


def _meas_noise_from_conf(conf: float, base_var: float, conf_floor: float) -> float:
    # Higher confidence => lower measurement noise. Clamp to [conf_floor,1].
    c = max(conf_floor, min(1.0, float(conf)))
    # Linear mapping: var = base_var / c
    return base_var / c


def distance_kalman(df: pd.DataFrame, cfg: KFConfig, m_per_px: Optional[float] = None) -> Dict[int, Dict[str, float]]:
    """Apply per-player Kalman smoothing on (cx,cy), then sum distance along the smoothed path.

    Gaps between frames are handled by propagating the model without updates.
    """
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]], dtype=float)
    I = np.eye(4)

    dt = 1.0 / max(1e-6, cfg.fps)
    A, Q = _kf_mats(dt, cfg.process_var)

    results: Dict[int, Dict[str, float]] = {}

    for pid, g in df.sort_values(["id", "frame"]).groupby("id"):
        g = g[["frame", "cx", "cy", "confidence"]].reset_index(drop=True)
        frames = g["frame"].to_numpy().astype(int)
        zs = g[["cx", "cy"]].to_numpy(dtype=float)
        confs = g["confidence"].to_numpy(dtype=float)

        # Initialize state from first measurement with zero velocity
        x = np.array([zs[0, 0], zs[0, 1], 0.0, 0.0], dtype=float)
        P = np.eye(4) * 1e3  # high uncertainty initially

        smoothed: List[Tuple[float, float]] = [(zs[0, 0], zs[0, 1])]

        for i in range(1, len(frames)):
            gap = int(frames[i] - frames[i-1])
            gap = max(1, gap)  # at least one step
            # Predict through the gap
            for _ in range(gap):
                x = A @ x
                P = A @ P @ A.T + Q
            # Update with measurement at current frame
            z = zs[i]
            R_var = _meas_noise_from_conf(confs[i], cfg.meas_var, cfg.conf_meas_floor)
            R = np.eye(2) * R_var
            y = z - (H @ x)               # innovation
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (I - K @ H) @ P
            smoothed.append((float(x[0]), float(x[1])))

        # Distance along smoothed positions only at measurement frames
        sm = np.array(smoothed)
        deltas = np.sqrt(np.diff(sm[:, 0])**2 + np.diff(sm[:, 1])**2)
        total_px = float(np.nansum(deltas))
        results[pid] = {
            "pixels": total_px,
            "meters": float(to_meters(total_px, m_per_px)),
        }

    return results


# -----------------------------
# 3) Optical Flow (Lucas–Kanade) distance
# -----------------------------

@dataclass
class OFConfig:
    win_size: Tuple[int, int] = (21, 21)
    max_level: int = 3
    criteria: Tuple[int, int, float] = (3 | 1, 30, 0.01)  # (type, maxCount, epsilon)
    reinit_on_detection: bool = True
    # Blend factor to fuse detection center with tracked point (0=trust flow only, 1=trust detection only)
    detection_blend: float = 0.3


def distance_optical_flow(
    video_path: str,
    df: pd.DataFrame,
    m_per_px: Optional[float] = None,
    of_cfg: OFConfig = OFConfig(),
) -> Dict[int, Dict[str, float]]:
    """Compute per-player distance by tracking bbox centers with LK optical flow.

    We initialize each player's track on its first detection, then propagate using
    calcOpticalFlowPyrLK frame-to-frame. When a fresh detection exists for the same
    frame, we optionally fuse it (or reinitialize) to reduce drift.
    """
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required for optical flow but not available.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    # Group detections by frame for quick access
    frames_group = {k: v for k, v in df.groupby("frame")}

    # Tracking state per player id
    pts: Dict[int, np.ndarray] = {}    # current point (1,1,2) float32 for LK API
    valid: Dict[int, bool] = {}        # whether point is currently valid
    totals_px: Dict[int, float] = {pid: 0.0 for pid in df["id"].unique()}

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        raise IOError("Failed to read first video frame.")
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_idx = 0

    # Initialize from detections at frame 0 if any
    if frame_idx in frames_group:
        g0 = frames_group[frame_idx]
        for _, row in g0.iterrows():
            pid = int(row["id"])
            pts[pid] = np.array([[[row["cx"], row["cy"]]]], dtype=np.float32)
            valid[pid] = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_idx += 1

        if not pts:
            # Initialize any new players that appear at this frame
            if frame_idx in frames_group:
                for _, row in frames_group[frame_idx].iterrows():
                    pid = int(row["id"])
                    pts[pid] = np.array([[[row["cx"], row["cy"]]]], dtype=np.float32)
                    valid[pid] = True
            prev_gray = gray
            continue

        # Prepare arrays for LK
        pids = [pid for pid, ok in valid.items() if ok]
        if pids:
            p0 = np.vstack([pts[pid] for pid in pids])  # (N,1,2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, p0, None,
                winSize=of_cfg.win_size,
                maxLevel=of_cfg.max_level,
                criteria=of_cfg.criteria,
            )
            for j, pid in enumerate(pids):
                if st[j] == 1:
                    old = p0[j, 0]
                    new = p1[j, 0]
                    move = float(np.hypot(*(new - old)))
                    totals_px[pid] += move
                    # Fuse with detection if available
                    if frame_idx in frames_group:
                        g = frames_group[frame_idx]
                        det = g[g["id"] == pid]
                        if len(det) > 0:
                            cx, cy = float(det.iloc[0]["cx"]), float(det.iloc[0]["cy"])
                            if of_cfg.reinit_on_detection:
                                new_pt = np.array([[[cx, cy]]], dtype=np.float32)
                            else:
                                new_pt = np.array([[[
                                    (1 - of_cfg.detection_blend) * new[0] + of_cfg.detection_blend * cx,
                                    (1 - of_cfg.detection_blend) * new[1] + of_cfg.detection_blend * cy,
                                ]]], dtype=np.float32)
                            pts[pid] = new_pt
                        else:
                            pts[pid] = np.array([[[new[0], new[1]]]], dtype=np.float32)
                    else:
                        pts[pid] = np.array([[[new[0], new[1]]]], dtype=np.float32)
                    valid[pid] = True
                else:
                    valid[pid] = False  # lost track; will try to reinit below
        # Reinitialize lost or missing tracks if detection exists at this frame
        if frame_idx in frames_group:
            g = frames_group[frame_idx]
            for _, row in g.iterrows():
                pid = int(row["id"])
                if pid not in valid or not valid[pid]:
                    pts[pid] = np.array([[[row["cx"], row["cy"]]]], dtype=np.float32)
                    valid[pid] = True

        prev_gray = gray

    cap.release()

    out: Dict[int, Dict[str, float]] = {}
    for pid, dist_px in totals_px.items():
        out[pid] = {
            "pixels": float(dist_px),
            "meters": float(to_meters(dist_px, m_per_px)),
        }
    return out


# -----------------------------
# Orchestration / CLI
# -----------------------------

def run_all(
    csv_path: str,
    fps: float = 25.0,
    m_per_px: Optional[float] = None,
    video_path: Optional[str] = None,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    df = load_detections(csv_path)
    # Keep only columns we need downstream
    df = df[["frame", "id", "x1", "y1", "x2", "y2", "confidence", "cx", "cy"]]

    eucl = distance_euclidean(df, m_per_px)
    kf = distance_kalman(df, KFConfig(fps=fps), m_per_px)

    results: Dict[str, Dict[int, Dict[str, float]]] = {
        "euclidean": eucl,
        "kalman": kf,
    }

    if video_path:
        of = distance_optical_flow(video_path, df, m_per_px)
        results["optical_flow"] = of

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute player travel distance with Euclidean, Kalman, and Optical Flow.")
    parser.add_argument("--csv", required=True, help="Path to detections CSV")
    parser.add_argument("--video", default=None, help="Path to the source video (required for optical flow)")
    parser.add_argument("--fps", type=float, default=25.0, help="Video FPS for Kalman dt")
    # parser.add_argument("--pixels-per-meter", type=float, default=None, help="Pixel density to convert distances to meters (px/m)")

    args = parser.parse_args()

    m_per_px = Scale("../data/data/points_cropped_schema.json").scale


    # Get the number fo fps of the given video
    def get_fps() -> float:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {args.video}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return float(fps)

    results = run_all(
        csv_path=args.csv,
        fps=get_fps() if args.video else args.fps,
        m_per_px=m_per_px,
        video_path=args.video,
    )

    # Pretty print
    def fmt(block: Dict[int, Dict[str, float]]) -> str:
        lines = []
        for pid in sorted(block.keys()):
            px = block[pid]["pixels"]
            m = block[pid]["meters"]
            if m_per_px:
                lines.append(f"  id {pid:>3}: {px:10.2f} px  |  {m:8.2f} m")
            else:
                lines.append(f"  id {pid:>3}: {px:10.2f} px")
        return "\n".join(lines)

    print("\nEuclidean distance:\n" + fmt(results["euclidean"]))
    print("\nKalman-smoothed distance:\n" + fmt(results["kalman"]))

    if "optical_flow" in results:
        print("\nOptical flow distance:\n" + fmt(results["optical_flow"]))
    else:
        print("\nOptical flow skipped (no --video provided).")


if __name__ == "__main__":
    main()
