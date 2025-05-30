from baseline_detection import BaselineDetection, warp_picture
from video_manager import VideoManager
import cv2


if __name__ == "__main__":
    print("START FIX VIDEO")

    # Init VideoManager
    vm = VideoManager(
        video_path="assets/extract-2.mp4",
        schema_path="assets/schema.png"
    )

    # Init baseline detection
    bd = BaselineDetection(
        schema=vm.schema,
        frame=vm.first_frame,
        frame_points_path="data/frame_points_fix.json",
        schema_points_path="data/schema_points_fix.json"
    )

    # Calculate homography between frame and schema
    h, h_inv = bd.calculate_homography()

    while vm.video.isOpened():
        ret, frame = vm.video.read()
        if not ret:
            print("[info]: End of video.")
            break

        # Apply baseline detection on the frame
        bd.frame = frame.copy()
        warped_frame = warp_picture(h=h, src=bd.frame, dest=bd.schema)

        # Identify lines
        lines_frame = bd.line_identification_full_court(warped_img=warped_frame)

        # Rewarp frame with detected lines
        inv_lines = warp_picture(h=h_inv, src=lines_frame, dest=bd.frame)
        # cv2.imwrite(filename="fix_inv.png", img=inv_lines)

        # Blend the lines with the original frame
        blended_frame = cv2.addWeighted(inv_lines, 0.7, bd.frame, 0.5, 0)

        # Resize warped_frame to match blended_frame dimensions if necessary
        if blended_frame.shape[1] != warped_frame.shape[1]:
            warped_frame = cv2.resize(warped_frame, (blended_frame.shape[1], blended_frame.shape[0]))

        # Ensure both frames have the same type
        if blended_frame.dtype != warped_frame.dtype:
            warped_frame = warped_frame.astype(blended_frame.dtype)

        combined_frame = cv2.vconcat([blended_frame, warped_frame])

        cv2.imshow("Frame", combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vm.video.release()
    cv2.destroyAllWindows()
