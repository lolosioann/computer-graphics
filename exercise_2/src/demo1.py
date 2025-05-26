import os
import numpy as np
import cv2
from render_object import render_object
from demo_utils import save_frame, create_video_from_frames, play_video_with_opencv


# Load Data and Parameters
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
road_center = data["k_road_center"]
cam_rel_pos = data["k_cam_car_rel_pos"]
up = data["k_cam_up"].flatten()

# Load texture image
tex_img = cv2.cvtColor(cv2.imread("stone-72_diffuse.jpg"), cv2.COLOR_BGR2RGB) / 255.0

# Output directory
output_dir = "demo_forward"
os.makedirs(output_dir, exist_ok=True)


def simulate_forward_looking_camera():
    global v_pos, v_uvs

    # Transpose if needed
    if v_pos.shape[0] == 3:
        v_pos = v_pos.T
    if v_uvs.shape[0] == 2:
        v_uvs = v_uvs.T

    v_colors = np.ones((v_pos.shape[0], 3), dtype=np.float32)  # Uniform white color 

    # Angular velocity 
    ang_vel = car_velocity / road_radius

    for i in range(num_frames):
        theta = i * ang_vel / fps  # Current angle (rads)

        # Compute car position and direction
        car_pos = road_center + road_radius * np.array([np.cos(theta), 0, np.sin(theta)])
        car_dir = np.array([-np.sin(theta), 0, np.cos(theta)])  # Tangent vector 

        # Compute camera coordinate frame (z forward, y up)
        z_cam = car_dir / np.linalg.norm(car_dir)
        x_cam = np.cross(up, z_cam)
        x_cam = x_cam / np.linalg.norm(x_cam)
        y_cam = np.cross(z_cam, x_cam)

        R = np.stack((x_cam, y_cam, z_cam), axis=1)  # World-to-camera rotation
        cam_pos = car_pos + R @ cam_rel_pos
        cam_target = cam_pos + z_cam  # Look forward in the direction of motion

        # Render the current frame
        img = render_object(
            v_pos, v_colors, t_pos_idx,
            sensor_height, sensor_width, res_h, res_w,
            focal, cam_pos, up, cam_target,
            uvs=v_uvs, texImg=tex_img
        )

        # Clip and convert to displayable format
        img = (img * 255.0).clip(0, 255)
        save_frame(img, i, output_dir)

if __name__ == "__main__":
    simulate_forward_looking_camera()
    create_video_from_frames(output_dir, "demo_forward.mp4", fps=25)
    play_video_with_opencv("demo_forward.mp4")
