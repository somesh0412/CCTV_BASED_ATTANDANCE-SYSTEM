from pathlib import Path
import cv2


# --------------------------------------------------
# Paths
# --------------------------------------------------

TESTING_DIR = Path(__file__).resolve().parent

VIDEO_DIR = TESTING_DIR / "videos"
GROUND_TRUTH_DIR = TESTING_DIR / "ground_truth"


# --------------------------------------------------
# Number of frames to extract from each video
# --------------------------------------------------

SAMPLES_PER_VIDEO = {
    "video1.mp4": 10,
    "video2.mp4": 20,
    "video3.mp4": 10,
}


# --------------------------------------------------
# Extract evenly spaced frames
# --------------------------------------------------

def extract_frames(video_name, sample_count):

    video_path = VIDEO_DIR / video_name

    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        return

    output_dir = GROUND_TRUTH_DIR / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"ERROR: Could not open {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        print(f"ERROR: No frames found in {video_name}")
        cap.release()
        return

    # Keep sample positions away from the very first/last frame
    frame_numbers = [
        int((i + 1) * total_frames / (sample_count + 1))
        for i in range(sample_count)
    ]

    print("\n" + "=" * 55)
    print(f"Video: {video_name}")
    print(f"Total frames: {total_frames}")
    print(f"Extracting: {sample_count} frames")
    print("=" * 55)

    for index, frame_number in enumerate(frame_numbers, start=1):

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        success, frame = cap.read()

        if not success:
            print(f"Could not read frame {frame_number}")
            continue

        output_path = output_dir / f"frame_{index:02d}.jpg"

        cv2.imwrite(
            str(output_path),
            frame
        )

        print(
            f"Saved {output_path.name} "
            f"(original frame {frame_number})"
        )

    cap.release()

    print(f"Saved frames to: {output_dir}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("\n==============================================")
    print("GROUND-TRUTH FRAME EXTRACTION")
    print("==============================================")

    for video_name, sample_count in SAMPLES_PER_VIDEO.items():
        extract_frames(video_name, sample_count)

    print("\n==============================================")
    print("FRAME EXTRACTION COMPLETED")
    print("==============================================")


if __name__ == "__main__":
    main()