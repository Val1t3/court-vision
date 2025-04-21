from components import BaselineDetection
import cv2
import numpy as np


if __name__ == "__main__":
    print("Video analysis")

    # TODO:
    # 1. Define importants points on the first frame
    # 2. Generate the tracking points on the first frame
    # 3. Use Optical Flow to track movement of the points on the second frame
    # 4. Implement the loop to apply this on each frame

    # TODO: Remake BaselineDetection object to include video instead of first frame,
    # include all the algo instead of inside the main.


    # 1.
    bd = BaselineDetection(
        video_path="assets/extract-1.mp4",
        schema_path="assets/schema.png",
        frame_points_path="data/frame_points.json",
        schema_points_path="data/schema_points.json"
    )

    h, h_inv = bd.calculate_homography()  # Calculate homography between frame and schema
    warped_frame = bd.warp_picture(h, bd.frame.copy(), bd.schema)  # Create warped frame
    lines_frame = bd.line_identification(warped_frame, 0)  # Apply line indentification on the warped frame
    lines_frame = bd.warp_picture(h_inv, lines_frame, bd.frame)  # Redimension the warped frame to the frame dimension
    cv2.imwrite("output/first_frame_lines.png", lines_frame)


    # 2.
    bd.generate_tracking_points()  # Generate points on the detected lines for the tracking management
    tracking_points = cv2.perspectiveTransform(bd.tracking_points.reshape(-1, 1, 2), h_inv)

    tracking_frame = bd.frame.copy()
    for pt in tracking_points:
        cv2.circle(tracking_frame, (int(pt[0][0]), int(pt[0][1])), 2, (0, 0, 255), -1)

    cv2.imwrite('output/tracking_points.png', tracking_frame)


    # 4.
    index = 0
    while bd.video.isOpened():
        # 3.
        prev_frame = bd.frame.copy()
        _, bd.frame = bd.video.read()

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(bd.frame, cv2.COLOR_BGR2GRAY)

        # Optical Flow
        p0 = tracking_points
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None)
        if p1 is None:
            raise ValueError(f"[Warning tracking points]: optical flow failed")

        # Filter valid points
        good_new = p1[st == 1]
        good_prev = p0[st == 1]
        if len(good_new) < 80:
            raise ValueError("[Filter valid points]: number of tracking points too small")

        show_frame = bd.frame.copy()
        for new, prev in zip(good_new, good_prev):
            x1, y1 = new.ravel()
            x2, y2 = prev.ravel()
            # print("x1, y1:", x1, y1)
            # print("x2, y2:", x2, y2)
            cv2.line(show_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
            cv2.circle(show_frame, (int(x1), int(y1)), 2, (0, 0, 255), -1)
            cv2.circle(show_frame, (int(x2), int(y2)), 2, (255, 0, 0), -1)

        # cv2.imwrite(f'output/optical_flow_{index}.png', show_frame)
        cv2.imshow('Frame', show_frame)
        index += 1
        p0 = good_new

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    bd.video.release()
    cv2.destroyAllWindows()
