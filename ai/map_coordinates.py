import cv2
import json
import sys

# Initialize global variables
points = []
window_name = "Point Mapping Tool"

def click_event(event, x, y, flags, param):
    """Handle mouse clicks to record points."""
    global points, image_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        # Record point
        points.append((x, y))

        # Draw a circle on the point
        cv2.circle(image_copy, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(image_copy, f"{len(points)}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.imshow(window_name, image_copy)

def save_points(points, output_file="points.json"):
    """Save recorded points to a JSON file."""
    with open(output_file, "w") as file:
        json.dump(points, file, indent=4)
    print(f"Points saved to {output_file}")

def load_points(input_file="points.json"):
    """Load previously saved points from a JSON file."""
    try:
        with open(input_file, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print("No previous points found. Starting fresh.")
        return []

def run_point_mapping_tool(image_path, output_file="ai/points.json"):
    """
    Run the point mapping tool to record points on an image.
    Press 's' to save the points and 'q' to quit without saving.

    Parameters
    ----------
    image_path : str
        Path to the image file to select points on.
    output_file : str
        Path to save the points to as a JSON file.
    """
    global image_copy, points

    # Load previous points if they exist
    points = load_points(output_file)

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read image file.")
        return

    # Show the image to select points
    image_copy = image.copy()
    cv2.imshow(window_name, image_copy)
    cv2.setMouseCallback(window_name, click_event)

    while True:
        cv2.imshow(window_name, image_copy)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):  # Save points
            save_points(points, output_file)
            break
        elif key == ord('q'):  # Quit without saving
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python map_coordinates.py <image_path>")
    else:
        run_point_mapping_tool(sys.argv[1])