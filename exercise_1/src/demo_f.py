import cv2
import numpy as np

import render_img

data = np.load("hw1.npy", allow_pickle=True).item()
print(data.keys())
# print(data["t_pos_idx"].shape)

# Read image
img = cv2.imread("fresque-saint-georges-2452226686.jpg")
# convert to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = render_img.render_img(
    faces=data["t_pos_idx"],
    vertices=data["v_pos2d"],
    vcolors=data["v_clr"],
    depth=data["depth"],
    shading="f",
    text_img=img,
)
img = (img * 255).clip(0, 255).astype(np.uint8)

# Convert RGB to BGR for OpenCV
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# Save and display image
cv2.imwrite("img.png", img)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
