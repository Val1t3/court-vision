from components import BaselineDetection
import matplotlib.pyplot as plt
import cv2


if __name__ == "__main__":
    print("Video analysis")

    # TODFO: FIRST FRAME SELECTION SIMULATION

    bd = BaselineDetection(
        frame_path="assets/test_image.png",
        schema_path="assets/schema.png",
        frame_points_path="data/frame_points.json",
        schema_points_path="data/schema_points.json"
    )

    h, h_inv = bd.calculate_homography()  # Calculate homography between frame and schema
    warped_res = bd.warp_picture(h, bd.frame, bd.schema)  # Create warped frame
    res = bd.line_identification(warped_res, 0)  # Apply line indentification on the warped frame
    bd.generate_tracking_points()  # Generate points on the detected lines for the tracking management

    # DON'T NEED TO SHOW THE POINTS
    # for pt in bd.tracking_points:
    #     cv2.circle(warped_res, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)

    first_frame = bd.warp_picture(h_inv, res, bd.frame)  # Warp frame with inv. homography
    cv2.imshow('Frame', first_frame)  # Display frame


    #------------------------------------------------------------------------#

    # VIDEO SIMULATION

    cap = cv2.VideoCapture('assets/extract-1.mp4')

    if (cap.isOpened() == False):
        raise ValueError("[Main video error] couldn't load video")

    # Read each frame of the video - working with 25fps
    while(cap.isOpened()):
        ret, frame = cap.read()  # Capture each frame
        if ret:
            # APPLY LINE IDENTIFICATION

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
