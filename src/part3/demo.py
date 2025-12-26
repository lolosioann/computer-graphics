from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image

from MatPhong import MatPhong
from render_object import render_object

# Get script directory
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "part3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the data dictionary from hw3.npy
data = np.load(SCRIPT_DIR / "hw3.npy", allow_pickle=True).item()

# Geometry data
v_pos = data["v_pos"]                # Vertex positions (3 x N)
v_uvs = data["v_uvs"]                # UV coordinates (N x 2)
t_pos_idx = data["t_pos_idx"].T      # Triangle indices (3 x Nt), transpose from (Nt x 3)

# Camera configuration
eye = data["cam_pos"].flatten()     # Camera position (3,)
target = data["target"].flatten()   # Camera target point (3,)
up = data["up"].flatten()           # Camera up vector (3,)
plane_h = data["plane_h"]           # Physical height of the camera plane
plane_w = data["plane_w"]           # Physical width of the camera plane
res_h = data["res_h"]               # Image resolution height
res_w = data["res_w"]               # Image resolution width
focal = data["focal"]               # Focal length of the camera

# Lighting configuration
l_pos = np.array(data["l_pos"])     # Light source positions (3 x 3)
l_int = np.array(data["l_int"])     # Light source intensities (3 x 3)
l_amb = data["l_amb"]               # Ambient light (3,)

# Phong exponent used in the "full" lighting mode
default_n = int(data["n"])

# Load texture image and normalize to [0,1]
tex_path = SCRIPT_DIR / "Mona-Lisa-Exist-in-Real-Life-2635825581.jpg"
texture = np.asarray(Image.open(tex_path)).astype(np.uint8) / 255.0

# Lighting configurations:
# Set only one component at a time for testing (ambient, diffuse, or specular),
# and a "full" mode that uses all three components from the data.
light_configs = {
    "ambient":  (data["ka"], 0.0, 0.0),
    "diffuse":  (0.0, data["kd"], 0.0),
    "specular": (0.0, 0.0, data["ks"]),
    "full":     (data["ka"], data["kd"], data["ks"])
}

# Main loop: render the object using both Gouraud and Phong shaders
# for each lighting configuration.
for shader in ["gouraud", "phong"]:
    for lighting_type, (ka, kd, ks) in light_configs.items():
        mat = MatPhong(ka=ka, kd=kd, ks=ks, n=default_n)

        img = render_object(
            v_pos=v_pos,
            v_uvs=v_uvs,
            t_pos_idx=t_pos_idx,
            tex=texture,
            plane_h=plane_h,
            plane_w=plane_w,
            res_h=res_h,
            res_w=res_w,
            focal=focal,
            eye=eye,
            up=up,
            target=target,
            mat=mat,
            l_pos=l_pos,
            l_int=l_int,
            l_amb=l_amb,
            shader=shader
        )

        # Save the rendered image
        filename = OUTPUT_DIR / f"render_{shader}_{lighting_type}.png"
        iio.imwrite(str(filename), (img * 255).astype(np.uint8))
        print(f"Saved: {filename}")
