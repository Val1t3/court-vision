# court-vision/src, player_detection_sahi.py
# Code written by Valentin Woehrel, 2025

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import visualize_object_predictions
import cv2
from typing import List
import csv

# constants
version = "11n"

const_video = "assets/extract-3.mp4"
const_model = "models/yolo" + version +".pt"
const_video_output = "output/player-detection-" + version + ".mp4"
const_results_path = "saves/player_detection_results.csv"
const_show = None
const_points = [
    [
        476,
        457
    ],
    [
        1442,
        470
    ],
    [
        129,
        670
    ],
    [
        1779,
        686
    ]
]


class ResultItem:
    """
    Class to save boxes at each frame.
    """

    def __init__(
            self,
            frame: int,
            id: int,
            x1: float,
            y1: float,
            x2: float,
            y2: float,
            confidence: float,
            type: int,
            # bbox: Optional[List[int]]
    ):
        self.frame = frame
        self.id = id
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.confidence = confidence
        self.type = type
        # self.bbox = bbox

    def to_dict(self):
        return {
            "frame": self.frame,
            "id": self.id,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "confidence": self.confidence,
            "type": self.type,
            # "bbox": self.bbox
        }


def player_detection_sahi(
        video: str,
        model: str,
        points: List[List[float]],
        video_output: str = None,
        show: bool = None,
        results_path: str = None,
    ) -> List[ResultItem]:
    """
    Function used to apply player detection using sliced detection and YOLO model.
    Returns a list of boxes coordinates on each frame.

    Parameters
    ----------
    video : str
        Path to the video to analyze.
    model : str
        Path to the YOLO model to use. If it doesn't exist, it will be
        automatically downloaded.
    points : str
        Coordinates of court points (Top-Left, Top-Right, Bottom-Left, Bottom-Right).
    video_output : str
        Path to save the output video. If nothing provided, don't save the video.
    results_path : str
        Path to save the results in a .csv file. If nothing provided, don't save
        the .csv file.
    show : bool
        Show the analyzed video in real time.

    Return
    ------
    List[]
        List of type {'frame': , 'id': , 'x1: , 'y1': , 'x2': , 'y2': ,
        'confidence': , 'class': }

    """
    results = []

    # create model detection
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model,
        confidence_threshold=0.3,
        device="cpu",
    )

    # open the input video
    cap = cv2.VideoCapture(video)

    # video writer to save the output (optional)
    if video_output != None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(video_output, fourcc, fps, (width, height))

    frame_index = 0

    # loop through frames
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            frame_index += 1
            break

        # predict using sliced prediction method
        result = get_sliced_prediction(
            frame,
            model,
            slice_height=800,
            slice_width=800,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            verbose=2,
        )

        # filter predictions to keep only "person" category
        result.object_prediction_list = [
            item for item in result.object_prediction_list
            if item.category.name == "person"
        ]

        # filter keeping boxes in court ares only
        if points:
            # margin = ((points[12][1] - points[10][1])
            #           + (points[4][1] - points[2][1])) / 2
            margin = 0

            # filter boxes not in the court area + margin
            filtered_pred = []
            for i in result.object_prediction_list:
                # convert bbox into 4 points
                bbox = i.bbox
                minx, miny, maxx, maxy = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy

                if (miny < max(points[2][1], points[3][1])
                    and maxy > min(points[0][1], points[2][0])
                    and minx < max(points[1][0], points[3][0])
                    and maxx > min(points[0][0], points[2][0])):
                    filtered_pred.append(i)

            result.object_prediction_list = filtered_pred

        # draw on image detected persons (optional)
        if video_output != None:
            res = visualize_object_predictions(
                image=frame,
                object_prediction_list=result.object_prediction_list,
                rect_th=2,
                text_size=1,
                text_th=1
            )

        for item in result.object_prediction_list:
            results.append(ResultItem(
                frame=frame_index,
                id=1,
                x1=item.bbox.minx,
                y1=item.bbox.miny,
                x2=item.bbox.maxx,
                y2=item.bbox.maxy,
                confidence=item.score.value,
                type=0,
                # bbox=item.bbox
            ))

        # save detected boxes in 'results' variable
        results.append(result.object_prediction_list)

        # show frame (optional)
        if show is True:
            cv2.imshow("SAHI Player Detection", res["image"])

        # write the frame to the output video (optional)
        if video_output != None:
            out.write(res["image"])

        # press 'q' to exit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # increment frame index
        frame_index += 1

    # release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if results_path != None:
        with open(results_path, "w", newline="") as csvfile:
            fieldnames = ["frame", "id", "x1", "y1", "x2", "y2", "confidence", "type"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for item in results:
                if isinstance(item, ResultItem):
                    writer.writerow(item.to_dict())
            print("Results available at", results_path)

    return results


if __name__ == "__main__":
    player_detection_sahi(
        video=const_video,
        model=const_model,
        video_output=const_video_output,
        results_path=const_results_path,
        show=const_show,
        points=const_points
    )
