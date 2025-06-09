from sahi import AutoDetectionModel
from sahi.predict import get_prediction
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import visualize_object_predictions
from PIL import Image


# Define constants
asset = "assets/frame.png"
model = "models/yolov11n.pt"


if __name__ == '__main__':
    # Create model detection
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model,
        confidence_threshold=0.3,
        device="cpu",
    )

    # Get width and height of the image
    with Image.open(asset) as img:
        width, height = img.size
        print(f"Image width: {width}, height: {height}")

    # Predict using sliced prediction method
    result = get_sliced_prediction(
        asset,
        detection_model,
        slice_height=1075,
        slice_width=1075,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )

    # Access the object prediction list
    object_prediction_list = result.object_prediction_list

    # # Export result
    result.export_visuals(export_dir="output/")
