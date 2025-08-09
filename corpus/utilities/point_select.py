# court-vision/src, point_select.py
# Code written by Valentin Woehrel, 2025

import json
import sys
import cv2


# Global variables
win_name: str = "Court Vision - Point Selection"
points: list = []
img: cv2.Mat | None = None
image: cv2.Mat | None = None
mouse_pos: list = [0, 0]
zoom_scale: float = 1.0
index = -1

def load(img_path: str, pts_path: str, idx: int = -1):
    """
    Load `image` and `points` files. Update `points`, `img` and `image`
    global variables.

    Parameters:
        img_path (str): Path to the 'image' file.
        pts_path (str): Path to the `points` file.

    """

    global img
    global index

    # Load index
    index = idx

    # Load image file
    img = cv2.imread(filename=img_path)
    if img is None:
        sys.exit("error: Could not read the image.")

    # Load points file
    try:
        with open(pts_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        with open(pts_path, "w", encoding="utf-8") as f:
            f.write("")
        data = None

    if data is not None:
        for i in data:
            points.append(i)
    else:
        print("no point to load")


def click_event(event: int, x: int, y: int, flags: int, param: any):
    """
    Callback function set to mouse events.

    Parameters:
        event (int): none
        x (int): X position
        y (int): Y position
        flags (int): none
        param (Any): none
    """

    global mouse_pos
    global index

    if event == cv2.EVENT_MOUSEMOVE:
        mouse_pos = [x, y]

    if event == cv2.EVENT_LBUTTONDOWN:
        if index != -1:
            points[index] = [x, y]
        else:
            points.append([x, y])


def draw_mouse_position():
    """
    Function to display on the image the actual position of the mouse.
    """

    global image

    text_pos = str(mouse_pos[0]) + ", " + str(mouse_pos[1])

    cv2.putText(
        img=image,
        text=text_pos,
        org=(50, 50),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 255, 0),  # BGR
        thickness=2,
        lineType=cv2.LINE_AA
    )


def draw_points():
    """
    Function to draw every item of `points` on the image with the id of each one.
    """

    for i, pt in enumerate(points):
        # Draw point
        cv2.circle(
            img=image,
            center=(pt[0], pt[1]),
            radius=5,
            color=(0, 0, 255),  # BGR
            thickness=-1
        )

        # Draw index
        cv2.putText(
            img=image,
            text=str(i),
            org=(pt[0] - 10, pt[1] - 20),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255),  # BGR
            thickness=2,
            lineType=cv2.LINE_AA
        )


def zoom_on_click():
    global image

    h, w = img.shape[:2]

    new_h = int(h * zoom_scale)
    new_w = int(w * zoom_scale)

    image = cv2.resize(src=image, dsize=(new_w, new_h))


def control_manager(pts_path: str) -> int:
    """
    Function to handle keyboard controls.

    Parameters:
        pts_path (str) : Path to the point file.

    Returns:
         int
    """

    global zoom_scale

    key = cv2.waitKey(1) & 0xFF

    # Remove last point
    if key == ord('z') and len(points) > 0:
        points.pop()
        return 0
    # Zoom in
    elif key == ord('i'):
        zoom_scale = min(float(zoom_scale + 0.1), 5.0)
        print("zoom in", zoom_scale)
        return 0
    # Zoom out
    elif key == ord('o'):
        zoom_scale = max(float(zoom_scale - 0.1), 1.0)
        print("zoom out", zoom_scale)
        return 0
    # Save
    elif key == ord('s'):
        with open(file=pts_path, mode="w") as f:
            json.dump(obj=points, fp=f, indent=4)
        print(f"Points saved to {pts_path}.")
        return 0
    # Quit program
    elif key == ord('q'):
        return 1
    # Default
    return 0


def help_manager():
    """
    Function to display help information to use the script.
    """

    print("Court Vision - Point Selection Tool")
    print("Run:")
    print("\tpython point_select.py `frame_path` `point_path`")
    print("Controls:")
    print("\tleft click: create new point")
    print("\tz: remove last point")
    print("\ti: zoom in")
    print("\to: zoom out")
    print("\tq: quit without saving")
    print("\ts: save and quit")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        help_manager()
        sys.exit(1)
    elif sys.argv[1] == "-h":
        help_manager()
        sys.exit()

    load(
        img_path=sys.argv[1],
        pts_path=sys.argv[2],
        idx=int(sys.argv[3]) if len(sys.argv) == 4 else -1
    )

    image = img.copy()
    cv2.imshow(winname=win_name, mat=image)
    cv2.setMouseCallback(win_name, click_event)

    while True:
        image = img.copy()

        zoom_on_click()
        draw_mouse_position()
        draw_points()
        cv2.imshow(winname=win_name, mat=image)

        if control_manager(pts_path=sys.argv[2]) == 1:
            break

    cv2.destroyAllWindows()
