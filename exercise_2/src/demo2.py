import os
import imageio.v2 as imageio
from render_object import render_object
import numpy as np
from demo_utils import save_frame, create_video_from_frames, play_video_with_opencv

# Load data
data = np.load("hw2.npy", allow_pickle=True).item()

# Rendering parameters
v_pos = data["v_pos"]
v_uvs = data["v_uvs"]
t_pos_idx = data["t_pos_idx"]
tex_img = imageio.imread("stone-72_diffuse.jpg") / 255.0

focal = data["k_f"]
plane_h = data["k_sensor_height"]
plane_w = data["k_sensor_width"]
res_h = 512
res_w = 512

fps = data["k_fps"]
duration = data["k_duration"] + 2
num_frames = int(duration * fps)

car_velocity = data["car_velocity"]
road_radius = data["k_road_radius"]
road_center = data["k_road_center"].flatten()
cam_rel_pos = data["k_cam_car_rel_pos"].flatten()
up = data["k_cam_up"].flatten()

output_dir = "demo_target"
os.makedirs(output_dir, exist_ok=True)

def simulate_camera_looking_at_target():
    global v_pos, v_uvs

    if v_pos.shape[0] == 3:
        v_pos = v_pos.T
    if v_uvs.shape[0] == 2:
        v_uvs = v_uvs.T

    v_colors = np.ones((v_pos.shape[0], 3), dtype=np.float32)

    ang_vel = car_velocity / road_radius
    cam_target = data["k_cam_target"].flatten()

    output_dir_target = "demo_target"
    os.makedirs(output_dir_target, exist_ok=True)

    for i in range(num_frames):
        theta = i * ang_vel / fps

        # Car position on circular path
        car_pos = road_center + road_radius * np.array([np.cos(theta), 0, np.sin(theta)])

        # Direction from camera to target
        z_cam = cam_target - (car_pos + cam_rel_pos)
        z_cam = z_cam / np.linalg.norm(z_cam)

        x_cam = np.cross(up, z_cam)
        x_cam = x_cam / np.linalg.norm(x_cam)
        y_cam = np.cross(z_cam, x_cam)

        R = np.stack((x_cam, y_cam, z_cam), axis=1)
        cam_pos = car_pos + R @ cam_rel_pos
        cam_pos = cam_pos.flatten()

        # Render with camera looking at fixed point
        img = render_object(
            v_pos, v_colors, t_pos_idx,
            plane_h, plane_w, res_h, res_w,
            focal, cam_pos, up, cam_target,
            uvs=v_uvs, texImg=tex_img
        )

        img = (img * 255.0).clip(0, 255)
        save_frame(img, i, output_dir_target)


if __name__ == "__main__":
    simulate_camera_looking_at_target()
    create_video_from_frames(output_dir, "demo_target.mp4", fps=25)
    play_video_with_opencv("demo_target.mp4")
