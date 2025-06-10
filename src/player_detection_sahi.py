from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import visualize_object_predictions
import cv2


# TODO:
# [X] Retrieve result object list
# [X] Filter with person category only
# [X] Draw boxes and texts on the image
# [ ] Check if box in the area of the court


# constants
asset = "assets/frame.png"
video = "assets/extract-3.mp4"
model = "models/yolov11n.pt"
output = "output/player-detection-v11n.mp4"
show = True


if __name__ == '__main__':
    # create model detection
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model,
        confidence_threshold=0.3,
        device="cpu",
    )

    # open the input video
    cap = cv2.VideoCapture(video)

    # Video writer to save the output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output, fourcc, fps, (width, height))

    # loop through frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
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

        # draw on image detected persons
        res = visualize_object_predictions(
            image=frame,
            object_prediction_list=result.object_prediction_list,
            rect_th=2,
            text_size=1,
            text_th=1
        )

        # show frame (optional)
        if show is True:
            cv2.imshow("SAHI Player Detection", res["image"])

        # write the frame to the output video
        out.write(res["image"])

        # press 'q' to exit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
