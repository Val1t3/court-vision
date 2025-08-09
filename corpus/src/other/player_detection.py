# court-vision/src, player_detection.py
# Code written by Valentin Woehrel, 2025

from ultralytics import YOLO
import cv2


if __name__ == '__main__':
    # Load yolo model
    model = YOLO("models/yolo8m.pt")

    # Open the input video
    cap = cv2.VideoCapture("assets/extract-3.mp4")

    # Video writer to save the output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter("output/extract-3.mp4", fourcc, fps, (width, height))

    # Loop through frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 on the frame
        results = model(frame)

        # Draw results on the frame
        annotated_frame = results[0].plot()

        # Show frame (optional)
        cv2.imshow("YOLOv8 Player Detection", annotated_frame)

        # Write the frame to the output video
        out.write(annotated_frame)

        # Press 'q' to exit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
