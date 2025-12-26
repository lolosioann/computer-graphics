
from typing import List, Union

import numpy as np

from light import light
from MatPhong import MatPhong

def shade_gouraud(
    v_pos: np.ndarray,                           # 3×3 projected triangle vertices in image space
    v_nrm: np.ndarray,                           # 3×3 normal vectors at triangle vertices
    v_uvs: np.ndarray,                           # 3×2 UV coordinates per vertex
    tex: np.ndarray,                             # texture image (H×W×3)
    cam_pos: np.ndarray,                         # camera position (3,)
    mat: MatPhong,                               # material
    l_pos: Union[np.ndarray, List[np.ndarray]],  # light positions
    l_int: Union[np.ndarray, List[np.ndarray]],  # light intensities
    l_amb: np.ndarray,                           # ambient light (3,)
    img: np.ndarray                              # image buffer to update
) -> np.ndarray:
    """
    Shade a triangle and update the specified image using Gouraud shading.
    """

    res_h, res_w, _ = img.shape

    # Compute color at each vertex
    # UV seam fix
    if np.max(v_uvs[:, 0]) - np.min(v_uvs[:, 0]) > 0.5:
        v_uvs = v_uvs.copy()
        for uv in v_uvs:
            if uv[0] < 0.5:
                uv[0] += 1.0

    vertex_colors = []
    for i in range(3):
        pt = v_pos[:, i]
        nrm = v_nrm[:, i]
        nrm = nrm / (np.linalg.norm(nrm) + 1e-8)
        uv = v_uvs[i]

        # Optional: wrapping
        uv[0] = uv[0] % 1.0
        uv[1] = np.clip(uv[1], 0, 1)

        # Sample texture
        tex_h, tex_w, _ = tex.shape
        u = np.clip(uv[0], 0, 1)
        v = np.clip(uv[1], 0, 1)
        tx = int(u * (tex_w - 1))
        ty = int((1 - v) * (tex_h - 1))

        vclr = tex[ty, tx, :] 

        color = light(pt, nrm, vclr, cam_pos, mat, l_pos, l_int, l_amb)
        vertex_colors.append(color)


    # Screen-space triangle (x, y)
    x = v_pos[0, :]
    y = v_pos[1, :]
    pts2D = np.stack([x, y], axis=1)

    # Bounding box
    min_x = max(int(np.floor(np.min(x))), 0)
    max_x = min(int(np.ceil(np.max(x))), res_w - 1)
    min_y = max(int(np.floor(np.min(y))), 0)
    max_y = min(int(np.ceil(np.max(y))), res_h - 1)

    # Rasterize
    for j in range(min_y, max_y + 1):
        for i in range(min_x, max_x + 1):
            p = np.array([i + 0.5, j + 0.5])

            A = np.array([
                [pts2D[0][0] - pts2D[2][0], pts2D[1][0] - pts2D[2][0]],
                [pts2D[0][1] - pts2D[2][1], pts2D[1][1] - pts2D[2][1]]
            ])
            b = p - pts2D[2]
            try:
                u, v = np.linalg.solve(A, b)
                w = 1 - u - v
            except np.linalg.LinAlgError:
                continue

            if u >= 0 and v >= 0 and w >= 0:
                color = u * vertex_colors[0] + v * vertex_colors[1] + w * vertex_colors[2]
                img[j, i, :] = np.clip(color, 0, 1)

    return img


# # Example usage of the shade_gouraud function (uncomment the following lines to test):

# # ----------------------------------------------------------------------------
# # DEMO 1: Strong lighting variation using different normals
# # ----------------------------------------------------------------------------

# import matplotlib.pyplot as plt
# from PIL import Image


# # Output image size
# res_h, res_w = 100, 100
# img = np.zeros((res_h, res_w, 3), dtype=np.float32) 

# # Create a solid red texture again
# tex = np.ones((64, 64, 3), dtype=np.uint8) * [255, 0, 0]

# # Projected triangle vertices (same as before)
# v_pos = np.array([
#     [20, 70, 45],
#     [20, 25, 80],
#     [0,  0,  0]
# ], dtype=np.float64)

# # Normals designed to simulate curved surface
# v_nrm = np.array([
#     [0,   0,   1],     # facing forward
#     [0.7, 0,   0.7],   # diagonal
#     [0,   0.7, 0.7]    # diagonal
# ], dtype=np.float64).T

# # Normalize the normals to unit length
# v_nrm = v_nrm / np.linalg.norm(v_nrm, axis=0)

# # UV mapping
# v_uvs = np.array([
#     [0.0, 1.0, 0.5],
#     [0.0, 0.0, 1.0]
# ], dtype=np.float64).T

# # Light and camera 
# cam_pos = np.array([45.0, 45.0, 100.0])
# l_pos = np.array([[45.0, 45.0, 100.0]])
# l_int = np.array([[1.0, 1.0, 1.0]])
# l_amb = np.array([0.0, 0.0, 0.0])  # no ambient

# # Material with strong specular component
# mat = MatPhong(ka=0.0, kd=0.002, ks=1.0, n=50)

# # Run shading
# img = shade_gouraud(v_pos, v_nrm, v_uvs, tex, cam_pos, mat, l_pos, l_int, l_amb, img)

# # Show result with visible gradient
# plt.imshow(img)
# plt.title("Gouraud Shading - Strong Lighting Gradient")
# plt.axis('off')
# plt.show()
