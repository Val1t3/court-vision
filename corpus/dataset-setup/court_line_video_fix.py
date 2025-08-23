# court-vision/src, court_line_video_fix.py
# Code written by Valentin Woehrel, 2025

from baseline_detection import BaselineDetection, warp_picture
from video_manager import VideoManager
import cv2
from enum import Enum


# constants
video_const = "../data/courtvision-dataset/eval_1.mov"
schema_const = "../data/assets/cropped_schema.png"

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

    # Calculate homography between frame and schema
    h, h_inv = bd.calculate_homography()



    # create video of schema points

    while vm.video.isOpened():
        ret, frame = vm.video.read()
        if not ret:
            print("[info]: End of video.")
            break

        # Apply baseline detection on the frame
        bd.frame = frame.copy()
        warped_frame = warp_picture(h=h, src=bd.frame, dest=bd.schema)

        ################################## LINE IDENTIFICATION
        # Identify lines
        # lines_frame = bd.line_identification_full_court(warped_img=warped_frame)

        # Rewarp frame with detected lines
        # inv_lines = warp_picture(h=h_inv, src=lines_frame, dest=bd.frame)
        ##################################


        # Blend the lines with the original frame
        blended_frame = cv2.addWeighted(bd.frame, 0.7, bd.frame, 0.5, 0)

        # Resize for concat
        warped_frame = cv2.resize(
            warped_frame, (blended_frame.shape[1], blended_frame.shape[0])
        )
        schema_frame = cv2.resize(
            vm.schema, (blended_frame.shape[1], blended_frame.shape[0])
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

    vm.video.release()
    cv2.destroyAllWindows()
