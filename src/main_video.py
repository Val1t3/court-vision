from baseline_detection import BaselineDetection
from court_tracker import CourtTracker
from video_manager import VideoManager
import cv2
import numpy as np


if __name__ == "__main__":
    print("START")

    # Init VideoManager
    vm = VideoManager(
        video_path="assets/extract-1.mp4",
        schema_path="assets/schema.png"
    )

    # Init BaselineDesction for the first_frame
    bd = BaselineDetection(
        schema=vm.schema,
        frame=vm.first_frame,
        frame_points_path="data/frame_points_2.json",
        schema_points_path="data/schema_points_2.json"
    )

    # Init CourtTracker
    tracker = CourtTracker(
        schema_points=bd.schema_points,
        side='right'
    )

    h, h_inv = tracker.calculate_homography_side(frame_points=bd.frame_points, schema_points=tracker.schema_pts, side='right')
    side = tracker.detect_visible_side(bd.frame.shape, h_inv)
    print(f"Detected side: {side}")

    # Init tracker.court_simulation
    simulation = np.array([
        [10000, 10000],
        [10000, 10000],
        [10000, 10000],
        [10000, 10000],
        [10000, 10000],
        [10000, 10000],
        [10000, 10000],
        [10000, 10000],
        [10000, 10000],
        [10000, 10000],
        [10000, 10000],
        [10000, 10000],

    ], dtype=np.float32)
    simulation = simulation.reshape(-1, 2)

    # Loop to analyze each frame
    while vm.video.isOpened():
        prev_frame = vm.frame.copy()
        ret, frame = vm.video.read()
        if not ret:
            print("[info]: End of video.")
            break

        show_frame = frame.copy()

        tracker.detect_visible_side(vm.frame.shape, h_inv)

        # Tracking Points manager
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        H, tracked_pts = tracker.track(prev_gray, frame_gray)
        print("TRACKED PTS:", tracked_pts)
        # if H is not None and tracked_pts is not None:
        #     for i, pt in enumerate(tracked_pts):
        #         x, y = int(pt[0]), int(pt[1])
        #         cv2.circle(show_frame, (x, y), 5, (0, 0, 255), -1)
        #         cv2.putText(show_frame, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        for i in range(6):
            simulation[6 + i] = tracked_pts[i]

        print("SIMULATION 1", simulation)

        simulation = tracker.simulate_pts(simulation, h_inv)

        print("SIMULATION 2", simulation)

        if H is not None and simulation is not None:
            for i, pt in enumerate(simulation):
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(show_frame, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(show_frame, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)


        # cv2.imwrite(f'output/optical_flow.png', show_frame)
        cv2.imshow('Frame', show_frame)
        print(f"Side: {tracker.current_side}")

        # break

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    vm.video.release()
    cv2.destroyAllWindows()

    print("END")
