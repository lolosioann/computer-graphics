import os
import numpy as np
import cv2 
from render_object import render_object
from demo_utils import save_frame, create_video_from_frames, play_video_with_opencv


data = np.load("hw2.npy", allow_pickle=True).item()


v_pos = data["v_pos"]           
v_uvs = data["v_uvs"]           
t_pos_idx = data["t_pos_idx"]   


focal = data["k_f"]
sensor_height = data["k_sensor_height"]
sensor_width = data["k_sensor_width"]
res_h = 512
res_w = 512


fps = data["k_fps"]
duration = data["k_duration"]
num_frames = int(duration * fps)

car_velocity = data["car_velocity"]
road_radius = data["k_road_radius"]
road_center = data["k_road_center"].flatten()
cam_rel_pos = data["k_cam_car_rel_pos"].flatten()
up = data["k_cam_up"].flatten()
cam_target = data["k_cam_target"].flatten()  


tex_img = cv2.cvtColor(cv2.imread("stone-72_diffuse.jpg"), cv2.COLOR_BGR2RGB) / 255.0

# Output directory
output_dir = "demo_target"
os.makedirs(output_dir, exist_ok=True)


def simulate_camera_looking_at_target():
    global v_pos, v_uvs

    # Ensure shape: (N, 3) and (N, 2)
    if v_pos.shape[0] == 3:
        v_pos = v_pos.T
    if v_uvs.shape[0] == 2:
        v_uvs = v_uvs.T

    # Uniform white color 
    v_colors = np.ones((v_pos.shape[0], 3), dtype=np.float32)

    # Angular velocity
    ang_vel = car_velocity / road_radius

    for i in range(num_frames):
        theta = i * ang_vel / fps  # rads

        # Compute car position on circular trajectory
        car_pos = road_center + road_radius * np.array([
            np.cos(theta), 0, np.sin(theta)
        ])

        # Compute camera view direction
        cam_world_pos = car_pos + cam_rel_pos
        z_cam = cam_target - cam_world_pos
        z_cam /= np.linalg.norm(z_cam)

        # Camera coordinate system
        x_cam = np.cross(up, z_cam)
        x_cam /= np.linalg.norm(x_cam)
        y_cam = np.cross(z_cam, x_cam)

        R = np.stack((x_cam, y_cam, z_cam), axis=1)
        cam_pos = car_pos + R @ cam_rel_pos

        # Render current frame
        img = render_object(
            v_pos, v_colors, t_pos_idx,
            sensor_height, sensor_width, res_h, res_w,
            focal, cam_pos, up, cam_target,
            uvs=v_uvs, texImg=tex_img
        )

        # Save frame to disk
        img = (img * 255.0).clip(0, 255)
        save_frame(img, i, output_dir)


if __name__ == "__main__":
    simulate_camera_looking_at_target()
    create_video_from_frames(output_dir, "demo_target.mp4", fps=25)
    play_video_with_opencv("demo_target.mp4")
