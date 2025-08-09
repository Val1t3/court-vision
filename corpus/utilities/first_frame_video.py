# court-vision/src, first_frame_video.py
# Code written by Valentin Woehrel, 2025

import cv2
import sys


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python first_frame_video.py <video_path>")
        sys.exit(1)

    # Open the video file
    video = cv2.VideoCapture(sys.argv[1])
    if not video.isOpened():
        print(f"Error: Unable to open video file at {sys.argv[1]}.")
        sys.exit(1)

    # Read the first frame
    ret, frame = video.read()
    if not ret:
        print("Error: Unable to grab a frame from the video.")
        sys.exit(1)

    # Save the first frame as an image
    cv2.imwrite("frame.png", frame)
    print("First frame saved as 'frame.png'.")

    video.release()
