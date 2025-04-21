import time

import cv2
import numpy as np

from render_img import render_img

# Load data
# Make sure to run the code in the correct directory
data = np.load("hw1.npy", allow_pickle=True).item()

# Read texture image
img = cv2.imread("texImg.jpg")

# Render image and time the process
time_start = time.time()
img = render_img(
    faces=data["t_pos_idx"],
    vertices=data["v_pos2d"],
    vcolors=data["v_clr"],
    uvs=data["v_uvs"],
    depth=data["depth"],
    shading="t",
    texImg=img,
)
time_end = time.time()
print("Rendering time:", time_end - time_start)

# Save image
img = (img * 255).clip(0, 255).astype(np.uint8)
cv2.imwrite("rendered_img_t.png", img)
