import json
import sys
import cv2


# Global variables
win_name = "Court Vision - Point Selection"
points = []
img = None
image = None
mouse_pos = [0, 0]


def load(img_path: str, pts_path: str):
    """
    Load `image` and `points` files. Update `points`, `img` and `image`
    global variables.

    Parameters:
        img_path (str): Path to the 'image' file.
        pts_path (str): Path to the `points` file.

    """

    global img

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

    mouse_pos = [x, y]

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        print(points)


def draw_mouse_position(x: int, y: int):
    """
    Function to display on the image the actual position of the mouse.

    Parameters:
        x (int): X position of the mouse
        y (int): Y position of the mouse
    """

    global image

    text_pos = str(x) + ", " + str(y)

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


def draw_points(point_list: list):
    """
    Function to display `points` on the image with the id of each one.

    Parameters:
        point_list (list): List of all points
    """

    for i, pt in enumerate(point_list):
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


def help_manager():
    # TODO: write content of help() function
    print("#####################################")
    print("Court Vision - Point Selection Tool")


if __name__ == '__main__':
    print(len(sys.argv))
    if sys.argv[1] == "-h":
        help_manager()
        sys.exit()
    elif len(sys.argv) != 3:
        help_manager()
        sys.exit(1)

    load(img_path=sys.argv[1], pts_path=sys.argv[2])

    image = img.copy()
    cv2.imshow(winname=win_name, mat=image)
    cv2.setMouseCallback(win_name, click_event)

    while True:
        image = img.copy()

        draw_mouse_position(x=mouse_pos[0], y=mouse_pos[1])
        draw_points(point_list=points)
        cv2.imshow(winname=win_name, mat=image)

        key = cv2.waitKey(1) & 0xFF

        # Remove last point
        if key == ord('z') and len(points) > 0:
            points.pop()

        # Quit program
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
