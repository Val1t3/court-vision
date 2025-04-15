import json
import numpy as np
import cv2


class BaselineDetection:
    """
    A clased used to detect baseline of basket court.

    Attributes
    ----------
    frame_pah : str
        Path to the frame file.
    schema_path : str
        Path to the schema file.
    frame_points_path : str
        Path to the frame points file.
    schema_points_path : str
        Path to the schema points file.
    frame : np.ndarray
        Frame image object.
    schema : np.ndarray
        Schema image object.
    """

    def __init__(self, frame_path: str, schema_path: str, frame_points_path: str, schema_points_path: str):
        """
        Initialize the BaselineDetection class.

        Parameters
        ----------
        frame_path : str
            Path to the frame file.
        schema_path : str
            Path to the schema file.
        frame_points_path : str
            Path to the frame points file.
        schema_points_path : str
            Path to the schema points file.
        """

        self.frame_path = frame_path
        self.schema_path = schema_path
        self.frame_points = None
        self.schema_points = None
        self.frame = None
        self.schema = None

        # Load images
        self.frame = cv2.imread(frame_path)
        self.schema = cv2.imread(schema_path)

        if self.frame is None or self.schema is None:
            raise ValueError("[BaselineDetection error]: couldn't load images")

        # Load points from JSON files
        with open(frame_points_path, 'r') as f:
            frame_data = json.load(f)
        with open(schema_points_path, 'r') as f:
            schema_data = json.load(f)

        if frame_data is None or schema_data is None:
            raise ValueError("[BaselineDetection error]: couldn't load points files")
        else:
            self.frame_points = np.array(frame_data, dtype=np.float32)
            self.schema_points = np.array(schema_data, dtype=np.float32)


    def calculate_homography(self) -> tuple:
        """
        Calculate the homography matrix from the frame points to the schema points.

        Returns
        -------
        np.ndarray
            The homography matrix.
        np.ndarray
            The inverse homography matrix.
        """

        # Check if points are in the correct format
        if len(self.frame_points) < 4 or len(self.schema_points) < 4:
            raise ValueError("[BaselineDetection error]: not enough points to calculate homography")

        # Calculate homography matrix
        h, _ = cv2.findHomography(self.frame_points, self.schema_points)
        h_inv, _ = cv2.findHomography(self.schema_points, self.frame_points)

        # Check if homography matrix is valid
        if h is None or h_inv is None:
            raise ValueError("[BaselineDetection error]: couldn't calculate homography matrix")

        return h, h_inv


    def warp_picture(self, h: np.ndarray, src: np.ndarray, dest: np.ndarray):
        """
        Warp the frame using the homography matrix.

        Returns
        -------
        np.ndarray
            The warped frame.
        """

        # Get the dimensions of the dest
        h_dest, w_dest = dest.shape[:2]

        # Warp the frame using the homography matrix
        warped_res = cv2.warpPerspective(src, h, (w_dest, h_dest))

        return warped_res


    def draw_line_between_points(self, image: np.ndarray, point1: list, point2: list) -> np.ndarray:
        """
        Draw a line between two points on the image.

        Parameters
        ----------
        image : np.ndarray
            The image on which to draw the line.
        point1 : list
            The first point (x, y).
        point2 : list
            The second point (x, y).

        Returns
        -------
        np.ndarray
            The image with the line drawn.
        """

        # Convert points to integers
        point1 = tuple(map(int, point1))
        point2 = tuple(map(int, point2))

        # Draw line on the image
        cv2.line(image, point1, point2, (0, 255, 0), 1)

        return image


    def line_identification(self, warped_img: np.ndarray) -> np.ndarray:
        """
        Identify lines in the warped image.

        Parameters
        ----------
        warped_img : np.ndarray
            The warped image.

        Returns
        -------
        np.ndarray
            The image with identified lines.
        """

        # Sideline
        res = self.draw_line_between_points(warped_img, self.schema_points[0], self.schema_points[1])
        res = self.draw_line_between_points(res, self.schema_points[1], self.schema_points[2])
        res = self.draw_line_between_points(res, self.schema_points[2], self.schema_points[3])
        res = self.draw_line_between_points(res, self.schema_points[3], self.schema_points[0])

        # 3-pts Line
        res = self.draw_line_between_points(res, self.schema_points[4], self.schema_points[5])
        res = self.draw_line_between_points(res, self.schema_points[6], self.schema_points[7])
        # Half Circle
        center_1 = (int((self.schema_points[4][0] + self.schema_points[7][0]) / 2),
                    int((self.schema_points[4][1] + self.schema_points[7][1]) / 2))  # Center between points 5 and 6
        center = (int((center_1[0] + self.schema_points[1][0]) / 2),
                  int(center_1[1]))  # Center of the basket position, between axis x of center_1 and point 2

        radius = int(np.linalg.norm(np.array(self.schema_points[4]) - np.array(self.schema_points[7])) / 2)
        radius = radius + 5  # Need explanations for the small offset incrementation

        cv2.ellipse(res, center, (radius, radius), 100, 0, 160, (0, 255, 0), 1)

        # Lane Line
        res = self.draw_line_between_points(res, self.schema_points[8], self.schema_points[9])
        res = self.draw_line_between_points(res, self.schema_points[10], self.schema_points[11])
        res = self.draw_line_between_points(res, self.schema_points[11], self.schema_points[8])

        return res