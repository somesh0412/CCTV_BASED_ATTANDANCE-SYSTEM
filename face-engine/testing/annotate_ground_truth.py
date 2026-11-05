from pathlib import Path
import cv2
import json


# --------------------------------------------------
# Paths
# --------------------------------------------------

TESTING_DIR = Path(__file__).resolve().parent

GROUND_TRUTH_DIR = TESTING_DIR / "ground_truth"
VIDEO1_DIR = GROUND_TRUTH_DIR / "video1"

OUTPUT_FILE = VIDEO1_DIR / "annotations.json"


# --------------------------------------------------
# Annotation state
# --------------------------------------------------

current_boxes = []
drawing = False
start_x = 0
start_y = 0


# --------------------------------------------------
# Mouse callback
# --------------------------------------------------

def mouse_callback(event, x, y, flags, param):

    global current_boxes
    global drawing
    global start_x
    global start_y

    if event == cv2.EVENT_LBUTTONDOWN:

        drawing = True
        start_x = x
        start_y = y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:

        display = param["image"].copy()

        # Draw existing boxes
        for box in current_boxes:
            x1, y1, x2, y2 = box

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

        # Draw current box
        cv2.rectangle(
            display,
            (start_x, start_y),
            (x, y),
            (0, 0, 255),
            2
        )

        cv2.imshow("Ground Truth Annotation", display)

    elif event == cv2.EVENT_LBUTTONUP:

        drawing = False

        x1 = min(start_x, x)
        y1 = min(start_y, y)

        x2 = max(start_x, x)
        y2 = max(start_y, y)

        # Ignore extremely small boxes
        if (x2 - x1) > 5 and (y2 - y1) > 5:

            current_boxes.append(
                [x1, y1, x2, y2]
            )


# --------------------------------------------------
# Annotate one image
# --------------------------------------------------

def annotate_image(image_path):

    global current_boxes

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_path}")
        return None

    current_boxes = []

    window_name = "Ground Truth Annotation"

    cv2.namedWindow(window_name)

    param = {
        "image": image
    }

    cv2.setMouseCallback(
        window_name,
        mouse_callback,
        param
    )

    while True:

        display = image.copy()

        # Draw existing boxes
        for i, box in enumerate(current_boxes):

            x1, y1, x2, y2 = box

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                display,
                f"Face {i + 1}",
                (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        cv2.putText(
            display,
            f"Faces marked: {len(current_boxes)}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display,
            "Draw boxes | U=undo | C=clear | ENTER=save | Q=quit",
            (10, display.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        cv2.imshow(window_name, display)

        key = cv2.waitKey(20) & 0xFF

        # Undo last box
        if key == ord("u"):

            if current_boxes:
                current_boxes.pop()

        # Clear all boxes
        elif key == ord("c"):

            current_boxes.clear()

        # Save this frame
        elif key == 13:  # ENTER

            cv2.destroyWindow(window_name)

            return current_boxes.copy()

        # Quit
        elif key == ord("q"):

            cv2.destroyWindow(window_name)

            return None


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("\n==============================================")
    print("VIDEO 1 GROUND-TRUTH ANNOTATION")
    print("==============================================")

    image_files = sorted(
        VIDEO1_DIR.glob("*.jpg")
    )

    if not image_files:

        print("\nNo JPG frames found.")

        print(
            f"Expected frames inside:\n{VIDEO1_DIR}"
        )

        return

    print(f"\nFound {len(image_files)} frames.")

    annotations = {}

    for index, image_path in enumerate(image_files, start=1):

        print("\n" + "-" * 50)
        print(f"Frame {index}/{len(image_files)}")
        print(f"File: {image_path.name}")

        print("\nDraw a box around EVERY visible face.")
        print("ENTER = save")
        print("U = undo")
        print("C = clear")
        print("Q = quit")

        boxes = annotate_image(image_path)

        if boxes is None:

            print("Annotation stopped.")

            break

        annotations[image_path.name] = boxes

        print(
            f"Saved {len(boxes)} ground-truth face(s)."
        )

    # --------------------------------------------------
    # Save annotations
    # --------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            annotations,
            file,
            indent=4
        )

    print("\n==============================================")
    print("ANNOTATION COMPLETED")
    print("==============================================")

    print(f"\nSaved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()