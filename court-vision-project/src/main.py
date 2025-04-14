from components import BaselineDetection
import matplotlib.pyplot as plt


if __name__ == "__main__":
    print("Hello Court Vision!")

    bd = BaselineDetection(
        frame_path="assets/test_image.png",
        schema_path="assets/schema.png",
        frame_points_path="data/frame_points.json",
        schema_points_path="data/schema_points.json"
    )

    h, h_inv = bd.calculate_homography()
    warped_res = bd.warp_picture(h, bd.frame, bd.schema)
    res = bd.line_identification(warped_res)
    new_res = bd.warp_picture(h_inv, res, bd.frame)

    plt.figure(figsize=(10, 10))
    plt.imshow(new_res, cmap='viridis')
    plt.axis("off")
    plt.show()
