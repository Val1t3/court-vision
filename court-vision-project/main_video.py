from components import BaselineDetection
import matplotlib.pyplot as plt
import cv2


if __name__ == "__main__":
    print("Video analysis")

    # FIRST FRAME SELECTION SIMULATION

    bd = BaselineDetection(
        frame_path="assets/test_image.png",
        schema_path="assets/schema.png",
        frame_points_path="data/frame_points.json",
        schema_points_path="data/schema_points.json"
    )

    #------------------------------------------------------------------------#

    # VIDEO SIMULATION

    cap = cv2.VideoCapture('assets/extract-1.mp4')

    if (cap.isOpened() == False):
        raise ValueError("[Main video error] couldn't load video")

    # Read each frame of the video - working with 25fps
    while(cap.isOpened()):
        ret, frame = cap.read()  # Capture each frame
        if ret:
            cv2.imshow('Frame', frame)  # Display frame

            # Close if 'Q' key pressed
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()  # Close all the frames

    #--------------------------------------------------------------------------#

    # # SAVE FIRST FRAME OF THE VIDEO
    # cap = cv2.VideoCapture('assets/extract-1.mp4')

    # if (cap.isOpened() == False):
    #     raise ValueError("[Main video error] couldn't load video")

    # ret, frame = cap.read()  # Capture each frame
    # if ret:
    #     cv2.imwrite('first_frame.png', frame)

    # cap.release()
    # cv2.destroyAllWindows()  # Close all the frames
