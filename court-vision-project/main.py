from components import BaselineDetection
import matplotlib.pyplot as plt


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
    res = bd.line_identification(warped_res, 0)  # Apply line indentification on the warped frame
    new_res = bd.warp_picture(h_inv, res, bd.frame)  # Warp frame with inv. homography

    plt.figure(figsize=(10, 10))
    plt.imshow(new_res, cmap='viridis')
    plt.axis("off")
    # plt.show()
    plt.savefig("output/main_fig.png")
