import time

import cv2
import numpy as np

import render_img

data = np.load("hw1.npy", allow_pickle=True).item()

# Read image
img = cv2.imread("texImg.jpg")

time_start = time.time()
img = render_img.render_img(
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
img = (img * 255).clip(0, 255).astype(np.uint8)

# Display image
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
