import cv2
import numpy as np
import imageio.v2 as imageio
import os

def save_frame(img: np.ndarray, idx: int, folder: str):
    img = (img * 255).astype(np.uint8)
    imageio.imwrite(f"{folder}/frame_{idx:03d}.png", img)

def create_video_from_frames(folder: str, output_file: str, fps: int, frame_size=(512, 512)):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'avc1' or 'XVID' if 'mp4v' fails
    video = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

    frame_files = sorted([
        f for f in os.listdir(folder) if f.endswith(".png")
    ])

    for filename in frame_files:
        frame_path = os.path.join(folder, filename)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Failed to load {frame_path}")
            continue
        video.write(frame)

    video.release()
    print(f"Video saved to {output_file}")

def play_video_with_opencv(video_path: str):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Rendered Video", frame)

        # Wait 40 ms per frame ~25 FPS
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def enforce_white_background(img, threshold=1e-3):
    mask = np.all(img < threshold, axis=-1)  # almost black
    img[mask] = 1.0  # set to white
    return img
