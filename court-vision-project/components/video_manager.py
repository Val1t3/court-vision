import cv2

class VideoManager:
    """
    A class used to managed images and video.

    Attributes
    ----------
    video : VideoCapture
    first_frame : np.ndarray
    schema : np.ndarray
    """

    def __init__(self, video_path: str, schema_path: str):
        self.video = None
        self.first_frame = None
        self.schema = None
        self.frame = None

        # Init self.video
        self.video = cv2.VideoCapture(video_path)
        if not self.video.isOpened():
            raise ValueError(f"[VideoManager error]: Unable to open video file at {video_path}.")

        # Init self.first_frame
        retval, frame = self.video.read()
        if not retval:
            raise ValueError("[VideoManager error]: Unable to grab a frame from the video.")
        self.first_frame = frame.copy()

        # Init self.frame
        self.frame = self.first_frame.copy()

        # Init self.schema
        self.schema = cv2.imread(schema_path)
        if self.schema is None:
            raise ValueError(f"[VideoManager error]: Unable to load schema image from {schema_path}.")