from baseline_detection import BaselineDetection
import matplotlib.pyplot as plt
import cv2


if __name__ == "__main__":
    print("Hello Court Vision!")

    bd = BaselineDetection(
        frame_path="assets/extract-1_first_frame.png",
        schema_path="assets/schema.png",
        frame_points_path="data/frame_points_1.json",
        schema_points_path="data/schema_points_1.json"
    )

    h, h_inv = bd.calculate_homography()  # Calculate homography between frame and schema
    warped_res = bd.warp_picture(h, bd.frame, bd.schema)  # Create warped frame
    res = bd.line_identification(warped_res, 'right')  # Apply line indentification on the warped frame
    bd.generate_tracking_points()  # Generate points on the detected lines for the tracking management
    for pt in bd.tracking_points:
        cv2.circle(warped_res, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)


    new_res = bd.warp_picture(h_inv, res, bd.frame)  # Warp frame with inv. homography



    plt.figure(figsize=(10, 10))
    plt.imshow(new_res, cmap='viridis')
    plt.axis("off")
    # plt.show()
    plt.savefig("output/main_fig.png")
