import cv2
import pandas as pd
import argparse
import sys

def draw_bbox_for_id(image_path, csv_path, target_id=1):
    """
    Draw bounding box for a specific ID on a given frame and display the image.

    Args:
        image_path (str): Path to the frame image
        csv_path (str): Path to the CSV file containing tracking data
        frame_number (int): Frame number to process
        target_id (int): ID to highlight (default: 1)
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Filter for the specific frame and ID
        frame_data = df[(df['id'] == target_id)]
        if frame_data.empty:
            print(f"No data found for ID {target_id}")
            return

        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return

        # Process each detection for the target ID in this frame
        for _, row in frame_data.iterrows():
            x1, y1, x2, y2, frame = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2']), int(row['frame'])
            confidence = row['confidence']

            print(f"Frame {frame}, ID {target_id}:")
            print(f"  Bounding box: ({x1}, {y1}) to ({x2}, {y2})")
            print(f"  Confidence: {confidence:.4f}")

            # Draw bounding box
            # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw circle
            center = (int((x1 + x2) / 2), int((y2)))
            cv2.circle(image, center, 5, (0, 255, 0), -1)

            # Add ID and confidence text
            # label = f"{frame}"
            # cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
            #            0.7, (0, 255, 0), 2)

        # Display the image
        window_name = f"ID {target_id}"
        cv2.imshow(window_name, image)
        print(f"\nDisplaying frame. Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Display bounding boxes for a specific ID on a frame')
    parser.add_argument('image_path', help='Path to the frame image')
    parser.add_argument('csv_path', help='Path to the CSV tracking file')
    # parser.add_argument('--frame_number', type=int, help='Frame number to process')
    parser.add_argument('--id', type=int, default=1, help='ID to highlight (default: 1)')

    args = parser.parse_args()

    draw_bbox_for_id(args.image_path, args.csv_path, args.id)

if __name__ == "__main__":
    # Example usage if run directly
    if len(sys.argv) == 1:
        print("Usage examples:")
        print("python script.py frame_001.jpg eval_1_tracks.csv 0")
        print("python script.py frame_010.jpg eval_1_tracks.csv 10 --id 2")
        print("\nOr call the function directly:")
        print("draw_bbox_for_id('frame_001.jpg', 'eval_1_tracks.csv', 0)")
    else:
        main()