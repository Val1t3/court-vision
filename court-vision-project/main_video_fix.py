from components import VideoManager, BaselineDetection
import cv2

if __name__ == "__main__":
    print("START FIX VIDEO")

    # Init VideoManager
    vm = VideoManager(
        video_path="assets/extract-2.mp4",
        schema_path="assets/schema.png"
    )

    # Init baseline detection thanks to the first frame and given points
    bd = BaselineDetection(
        schema=vm.schema,
        frame=vm.first_frame,
        frame_points_path="data/frame_points_fix.json",
        schema_points_path="data/schema_points_fix.json"
    )

    # Calculate homography between frame and schema
    h, h_inv = bd.calculate_homography()
    warped_frame = bd.warp_picture(h=h, src=bd.frame, dest=bd.schema)

    # Identify lines
    lines_frame = bd.line_identification_full_court(warped_img=warped_frame)
    cv2.imwrite(filename="fix.png", img=lines_frame)

    # Rewarp frame with detected lines
    inv_lines = bd.warp_picture(h=h_inv, src=lines_frame, dest=bd.frame)
    cv2.imwrite(filename="fix_inv.png", img=inv_lines)

    # Need to apply on each frame of the video