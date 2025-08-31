# court-vision/src, court_line_video_fix.py
# Code written by Valentin Woehrel, 2025

from baseline_detection import BaselineDetection, warp_picture
from video_manager import VideoManager
from schema_video import create_video
import pandas as pd
import cv2
from enum import Enum


# constants
name = 'eval_2'

video_const = "../data/courtvision-dataset/" + name + ".mov"
schema_const = "../data/assets/cropped_schema.png"

coordinates_path = "../data/courtvision-dataset/" + name + "_tracks.csv"

euclidean_path = "../data/courtvision-dataset/" + name + "_dist_euclidean.csv"
kalman_path = "../data/courtvision-dataset/" + name + "_dist_kalman.csv"
opticalflow_path = "../data/courtvision-dataset/" + name + "_dist_optical_flow.csv"

frame_points_const = "../data/data/eval_points.json"
schema_points_const = "../data/data/points_cropped_schema.json"


class Show(Enum):
    """
    Class used to choose what type of analysis to display.
    """

    ALL = 0  # Show all
    BLENDED = 1  # Show video with lines
    WARPED = 2  # Show warped video to schema with lines
    SCHEMA = 3  # Show schema reproduction


if __name__ == "__main__":
    print("START FIX VIDEO")

    # Constant
    show = Show.ALL
    win_name = "Court Vision - Baseline Detection"

    # Init VideoManager
    vm = VideoManager(video_path=video_const, schema_path=schema_const)

    # Init BaselineDetection
    bd = BaselineDetection(
        schema=vm.schema,
        frame=vm.first_frame,
        frame_points_path=frame_points_const,
        schema_points_path=schema_points_const,
    )

    # calculate homography between frame and schema
    h, h_inv = bd.calculate_homography()

    # retrieve cum distance for each algo
    euclidean_cum_m = pd.read_csv(euclidean_path)
    kalman_cum_m = pd.read_csv(kalman_path)
    opticalflow_cum_m = pd.read_csv(opticalflow_path)

    max_frame = euclidean_cum_m["frame"].max()

    # create video of schema points
    create_video(name)

    sch_video = cv2.VideoCapture('../data/output/schema_position_' + name + '.mp4')

    coordinates_df = pd.read_csv(coordinates_path)

    frame_num = 0

    while vm.video.isOpened() or frame_num < max_frame:
        ret, frame = vm.video.read()
        sch_ret, sch_frame = sch_video.read()
        if not ret or not sch_ret:
            print("[info]: End of video.")
            break

        # Apply baseline detection on the frame
        bd.frame = frame.copy()
        warped_frame = warp_picture(h=h, src=bd.frame, dest=bd.schema)

        ##################################
        # Identify lines
        # lines_frame = bd.line_identification_full_court(warped_img=warped_frame)

        # Rewarp frame with detected lines
        # inv_lines = warp_picture(h=h_inv, src=lines_frame, dest=bd.frame)
        ##################################

        # distances
        euclidean_vals = euclidean_cum_m[euclidean_cum_m['frame'] == frame_num]['cum_m'].values
        kalman_vals = kalman_cum_m[kalman_cum_m['frame'] == frame_num]['cum_m'].values
        opticalflow_vals = opticalflow_cum_m[opticalflow_cum_m['frame'] == frame_num]['cum_m'].values

        euclidean_dist = euclidean_vals[0] if len(euclidean_vals) > 0 else euclidean_cum_m['cum_m'].values[-1]
        kalman_dist = kalman_vals[0] if len(kalman_vals) > 0 else kalman_cum_m['cum_m'].values[-1]
        opticalflow_dist = opticalflow_vals[0] if len(opticalflow_vals) > 0 else opticalflow_cum_m['cum_m'].values[-1]

        # write cum distance for each algo
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        thickness = 2

        cv2.putText(
            img=sch_frame,
            text=f"Euclidean: {euclidean_dist:.2f} m",
            fontFace=font,
            fontScale=font_scale,
            color=(255, 255, 0),
            thickness=thickness,
            org=(0, 20)
        )

        cv2.putText(
            img=sch_frame,
            text=f"Kalman: {kalman_dist:.2f} m",
            fontFace=font,
            fontScale=font_scale,
            color=(0, 255, 80),
            thickness=thickness,
            org=(0, 40)
        )

        cv2.putText(
            img=sch_frame,
            text=f"Optical Flow: {opticalflow_dist:.2f} m",
            fontFace=font,
            fontScale=font_scale,
            color=(0, 100, 255),
            thickness=thickness,
            org=(0, 60)
        )


        # Draw trajectory - draw position on each frame before frame_num
        for i in range(frame_num):
            frame_coo = coordinates_df[(coordinates_df["frame"] == i) & (coordinates_df["id"] == 1)]
            center = (int((frame_coo["x1"].values[0] + frame_coo["x2"].values[0]) / 2), int(frame_coo["y2"].values[0]))
            cv2.circle(bd.frame, center, 15, (0, 0, 255), -1)

        # Blend the lines with the original frame
        blended_frame = cv2.addWeighted(bd.frame, 0.7, bd.frame, 0.5, 0)

        # Resize for concat
        warped_frame = cv2.resize(
            warped_frame, (blended_frame.shape[1], blended_frame.shape[0])
        )
        schema_frame = cv2.resize(
            sch_frame, (blended_frame.shape[1], blended_frame.shape[0])
        )

        # Ensure both frames have the same type
        if blended_frame.dtype != warped_frame.dtype:
            warped_frame = warped_frame.astype(blended_frame.dtype)
        if blended_frame.dtype != schema_frame.dtype:
            schema_frame = schema_frame.astype(blended_frame.dtype)

        # What displays on screen
        if show == Show.ALL:
            combined_frame = cv2.vconcat([blended_frame, warped_frame, schema_frame])
            cv2.imshow(win_name, combined_frame)
        elif show == Show.BLENDED:
            cv2.imshow(win_name, blended_frame)
        elif show == Show.WARPED:
            cv2.imshow(win_name, warped_frame)
        elif show == Show.SCHEMA:
            cv2.imshow(win_name, schema_frame)

        # Control Manager
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            cv2.imwrite("output/screen.png", warped_frame)
            print("take screenshot")

        frame_num += 1

    vm.video.release()
    cv2.destroyAllWindows()
