from components import BaselineDetection, CourtTracker
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
        frame_points_path="data/frame_points_2.json",
        schema_points_path="data/schema_points_2.json"
    )


    # h, h_inv = bd.calculate_homography()  # Calculate homography between first frame and schema
    # warped_frame = bd.warp_picture(h, bd.frame.copy(), bd.schema)  # Create warped frame
    # lines_frame = bd.line_identification(warped_frame, 'right')  # Apply line indentification on the warped frame
    # lines_frame = bd.warp_picture(h_inv, lines_frame, bd.frame)  # Redimension the warped frame to the frame dimension
    # cv2.imwrite("output/first_frame_lines.png", lines_frame)

    tracker = CourtTracker(schema_points=bd.schema_points, side='right')
    h, h_inv = tracker.calculate_homography_side(frame_points=bd.frame_points, schema_points=tracker.schema_pts, side='right')
    side = tracker.detect_visible_side(bd.frame.shape, h_inv)
    print(f"Detected side: {side}")


    # 2.
    # bd.generate_tracking_points()  # Generate points on the detected lines for the tracking management
    # tracking_points = cv2.perspectiveTransform(bd.tracking_points.reshape(-1, 1, 2), h_inv)

    # tracking_frame = bd.frame.copy()
    # for pt in tracking_points:
    #     cv2.circle(tracking_frame, (int(pt[0][0]), int(pt[0][1])), 2, (0, 0, 255), -1)

    # cv2.imwrite("output/tracking_points.png", tracking_frame)

    # 4.
    while bd.video.isOpened():
        # 3.
        prev_frame = bd.frame.copy()
        ret, frame = bd.video.read()
        if not ret:
            print("[info]: End of video.")
            break

        bd.frame = frame
        show_frame = bd.frame.copy()

        tracker.detect_visible_side(bd.frame.shape, h_inv)

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(bd.frame, cv2.COLOR_BGR2GRAY)

        H, tracked_pts = tracker.track(prev_gray, frame_gray)
        if H is not None and tracked_pts is not None:
            for i, pt in enumerate(tracked_pts):
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(show_frame, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(show_frame, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # cv2.imwrite(f'output/optical_flow_{index}.png', show_frame)
        cv2.imshow('Frame', show_frame)
        print(f"Side: {tracker.current_side}")

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    bd.video.release()
    cv2.destroyAllWindows()
