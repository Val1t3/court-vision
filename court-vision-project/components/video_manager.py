import cv2
import numpy as np

class VideoManager:
    """
    A class used to managed images and video.

    Attributes
    ----------
    video : VideoCapture
    frame : np.ndarray
    first_frame : np.ndarray
    schema : np.ndarray
    """
    # Do we remove `self.first_frame`?


    def __init__(self, video_path: str, schema_path: str):
        self.video = None
        self.frame = None
        self.first_frame = None
        self.schema = None

        self.video = cv2.VideoCapture(video_path)
        if not self.video.isOpened():
            raise ValueError(f"[VideoManager error]: Unable to open video file at {video_path}.")

        self.frame = self._get_frame()
        self.first_frame = self.frame.copy()

        self.schema = cv2.imread(schema_path)
        if self.schema is None:
            raise ValueError(f"[VideoManager error]: Unable to load schema image from {schema_path}.")


    def _get_frame(self) -> np.ndarray:
        retval, frame = self.video.read()
        if not retval:
            raise ValueError("[VideoManager error]: Unable to grab a frame from the video.")
        if frame is None:
            raise ValueError("[VideoManager error]: The grabbed frame is empty.")
        return frame
