import time
from pathlib import Path

import cv2
import numpy as np

from render_img import render_img

# Get script directory
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "part1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
data = np.load(SCRIPT_DIR / "hw1.npy", allow_pickle=True).item()

# Read texture image
img = cv2.imread(str(SCRIPT_DIR / "texImg.jpg"))

# Render image and time the process
time_start = time.time()
img = render_img(
    faces=data["t_pos_idx"],
    vertices=data["v_pos2d"],
    vcolors=data["v_clr"],
    uvs=data["v_uvs"],
    depth=data["depth"],
    shading="f",
    texImg=img,
)
time_end = time.time()
print("Rendering time:", time_end - time_start)

# Save image
img = (img * 255).clip(0, 255).astype(np.uint8)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite(str(OUTPUT_DIR / "rendered_img_f.png"), img)
