import os
import cv2
import numpy as np

def save_frame(img: np.ndarray, idx: int, folder: str):
    """
    Saves a single image frame to disk.

    Parameters:
        img (np.ndarray): The image to save, expected in [0, 1] float32 format.
        idx (int): The frame index used for filename formatting.
        folder (str): Directory to save the frame.
    """
    os.makedirs(folder, exist_ok=True)
    img_uint8 = (img * 255).astype(np.uint8)
    filename = os.path.join(folder, f"frame_{idx:03d}.png")
    cv2.imwrite(filename, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))


def create_video_from_frames(folder: str, output_file: str, fps: int, frame_size=(512, 512)):
    """
    Compiles a series of image frames into an MP4 video using OpenCV.

    Parameters:
        folder (str): Directory containing the image frames.
        output_file (str): Output path for the resulting video.
        fps (int): Frames per second.
        frame_size (tuple): Frame resolution (width, height).
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

    frame_files = sorted(
        f for f in os.listdir(folder) if f.lower().endswith(".png")
    )

    for filename in frame_files:
        frame_path = os.path.join(folder, filename)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"[Warning] Failed to load frame: {frame_path}")
            continue
        resized = cv2.resize(frame, frame_size)
        video.write(resized)

    video.release()
    print(f"[Info] Video saved to {output_file}")


def play_video_with_opencv(video_path: str):
    """
    Plays a video using OpenCV's imshow window.

    Parameters:
        video_path (str): Path to the video file.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[Error] Could not open video: {video_path}")
        return

    print("[Info] Playing video. Press 'q' to quit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Rendered Video", frame)

        # Press 'q' to exit early
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
