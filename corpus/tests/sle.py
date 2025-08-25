import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Iterable, Optional, Tuple

def segment_length_error_sle(
    csv_euclidean_path: str,
    csv_kalman_path: str,
    csv_opticalflow_path: str,
    player_id: int,
    true_length_m: float,
    methods: Optional[Iterable[str]] = None,
    plot: bool = True,
    title: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[plt.Figure]]:
    """
    Segment Length Error (SLE) on Known Lines.

    Computes, for each method, the distance traveled by `player_id` between
    [start_frame, end_frame] by summing `step_m`, then compares to the known
    segment length L to obtain Absolute and Relative errors.

    Parameters
    ----------
    csv_path : str
        Path to CSV with columns: frame,id,step_m,cum_m,method
    player_id : int
        The player's tracking ID to evaluate.
    start_frame : int
        Inclusive start frame index for the segment.
    end_frame : int
        Inclusive end frame index for the segment (must be >= start_frame).
    true_length_m : float
        Ground-truth segment length L in meters (must be > 0).
    methods : Optional[Iterable[str]]
        Subset of methods to include (e.g., ["euclidean","kalman","optical_flow"]).
        If None, all methods present in the file are used.
    plot : bool
        If True, render a per-method bar chart of Relative Error.
    title : Optional[str]
        Custom plot title.

    Returns
    -------
    results_df : pandas.DataFrame
        Columns: method, frames_used, D_hat_m, L_true_m, AbsErr_m, RelErr,
                 start_frame, end_frame, player_id
    fig : Optional[matplotlib.figure.Figure]
        The figure object if plot=True, else None.

    Notes
    -----
    - Uses sum of `step_m` within the frame window to avoid issues if `cum_m`
      resets or tracks are fragmented.
    - If frames are missing inside the window, only present frames contribute.
    """
    # --- Validations
    if true_length_m <= 0:
        raise ValueError("true_length_m must be > 0.")

    csv_euclidean_path = Path(csv_euclidean_path)
    if not csv_euclidean_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_euclidean_path}")

    csv_kalman_path = Path(csv_kalman_path)
    if not csv_kalman_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_kalman_path}")

    csv_opticalflow_path = Path(csv_opticalflow_path)
    if not csv_opticalflow_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_opticalflow_path}")

    # --- Load and basic checks
    df_euclidean = pd.read_csv(csv_euclidean_path)
    df_kalman = pd.read_csv(csv_kalman_path)
    df_opticalflow = pd.read_csv(csv_opticalflow_path)

    required_cols = {"frame", "id", "step_m", "cum_m", "method"}
    missing = required_cols - set(df_euclidean.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    # --- Combine all method DataFrames
    df = pd.concat([df_euclidean, df_kalman, df_opticalflow], ignore_index=True)

    # --- Define start_frame with 0 and end_frame with the largest frame value in the CSVs
    start_frame = 0.0
    max_frame = df["frame"].max()
    end_frame = max_frame

    if end_frame < start_frame:
        raise ValueError("end_frame must be >= start_frame.")

    # --- Filter to player and frame window
    w = (df["id"] == player_id) & (df["frame"] >= start_frame) & (df["frame"] <= end_frame)
    dfw = df.loc[w].copy()
    if dfw.empty:
        raise ValueError(
            f"No rows for player_id={player_id} within frames [{start_frame}, {end_frame}]."
        )

    if methods is not None:
        methods = {str(m) for m in methods}
        dfw = dfw[dfw["method"].isin(methods)]
        if dfw.empty:
            raise ValueError(f"No rows after filtering to methods={methods}.")

    # --- Aggregate distance per method (sum of step_m)
    agg = (
        dfw.groupby("method", sort=True)
           .agg(D_hat_m=("step_m", "sum"), frames_used=("frame", "nunique"))
           .reset_index()
    )

    # Compute errors
    agg["L_true_m"] = float(true_length_m)
    agg["AbsErr_m"] = (agg["D_hat_m"] - agg["L_true_m"]).abs()
    agg["RelErr"] = agg["AbsErr_m"] / agg["L_true_m"]
    agg["start_frame"] = int(start_frame)
    agg["end_frame"] = int(end_frame)
    agg["player_id"] = int(player_id)

    # --- Plot
    fig = None
    if plot:
        fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=140)
        ax.bar(agg["method"], agg["RelErr"], color="#4c78a8")
        for i, v in enumerate(agg["RelErr"].values):
            ax.text(i, v + max(0.005, 0.02 * max(agg["RelErr"].max(), 1e-6)),
                    f"{v:.2%}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Relative Error (|D̂ - L| / L)")
        ttl = title or f"SLE on Known Line · Player {player_id} · Frames [{start_frame}, {end_frame}]"
        ax.set_title(ttl)
        ax.set_ylim(0, max(agg["RelErr"].max() * 1.25, 0.05))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_xlabel("Method")
        ax.set_axisbelow(True)
        fig.tight_layout()

    return agg, fig


if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser(
    #     description="Execute SLE test to comapre all distance calculation algorithms."
    # )
    # parser.add_argument("--name", type=str, help="Name of the eval.")
    # parser.add_argument("--player-id", type=int, help="Id of the player to do the test.")
    # parser.add_argument("--start", type=int, help="Start frame.")
    # parser.add_argument("--end", type=int, help="End frame.")
    # parser.add_argument("--length", type=float, help="True length in meters.")


    # args = parser.parse_args()

    name = "eval_1"
    true_length = 15

    res, fig = segment_length_error_sle(
        csv_euclidean_path=f"../data/courtvision-dataset/{name}_dist_euclidean.csv",
        csv_kalman_path=f"../data/courtvision-dataset/{name}_dist_kalman.csv",
        csv_opticalflow_path=f"../data/courtvision-dataset/{name}_dist_optical_flow.csv",
        player_id=1,
        true_length_m=15,
        methods=None,
        plot=True,
        title=f"SLE Comparison · Player 1 · {name} · True Length: {true_length}m"
    )
    print(res)
    fig.savefig(f"{name}_sle_plot_.png")
