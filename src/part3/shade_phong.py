from typing import List, Union

import numpy as np

from light import light
from MatPhong import MatPhong

def shade_phong(
    v_pos: np.ndarray,                               # 3x3 triangle vertices in image space (after projection)
    v_nrm: np.ndarray,                               # 3x3 vertex normals
    v_uvs: np.ndarray,                               # 3x2 (u,v) texture coordinates per vertex
    tex: np.ndarray,                                 # texture image (H x W x 3)
    cam_pos: np.ndarray,                             # camera/viewer position (3,)
    mat: MatPhong,                                   # Phong material
    l_pos: Union[np.ndarray, List[np.ndarray]],      # light positions (N, 3)
    l_int: Union[np.ndarray, List[np.ndarray]],      # light intensities (N, 3)
    l_amb: np.ndarray,                               # ambient light (3,)
    img: np.ndarray                                  # image buffer to update (H x W x 3)
) -> np.ndarray:
    """
    Phong shading: interpolate normals and UVs per pixel, but reuse fixed V and L per triangle.
    Lighting is calculated per-pixel with interpolated normals, using fixed V and L from triangle centroid.
    """
    res_h, res_w, _ = img.shape

    # Extract 2D screen coordinates
    x = v_pos[0, :]
    y = v_pos[1, :]
    pts2D = np.stack([x, y], axis=1)

    # Bounding box of triangle (clamped to image bounds)
    min_x = max(int(np.floor(np.min(x))), 0)
    max_x = min(int(np.ceil(np.max(x))), res_w - 1)
    min_y = max(int(np.floor(np.min(y))), 0)
    max_y = min(int(np.ceil(np.max(y))), res_h - 1)

    # Compute triangle centroid (in 3D, before projection)
    pt_center = np.mean(v_pos, axis=1)

    # Fixed view direction V
    V = cam_pos - pt_center
    V = V / (np.linalg.norm(V) + 1e-8)

    # Fixed light directions for all light sources
    L_list = l_pos - pt_center
    L_list = L_list / (np.linalg.norm(L_list, axis=1, keepdims=True) + 1e-8)

    # Rasterization over bounding box
    for j in range(min_y, max_y + 1):
        for i in range(min_x, max_x + 1):
            p = np.array([i + 0.5, j + 0.5])  # Pixel center

            # Compute barycentric coordinates
            A = np.array([
                [pts2D[0][0] - pts2D[2][0], pts2D[1][0] - pts2D[2][0]],
                [pts2D[0][1] - pts2D[2][1], pts2D[1][1] - pts2D[2][1]]
            ])
            b = p - pts2D[2]
            try:
                u, v = np.linalg.solve(A, b)
                w = 1 - u - v
            except np.linalg.LinAlgError:
                continue  # Degenerate triangle
            
            # Check if inside triangle
            if u >= 0 and v >= 0 and w >= 0:
                
             
                # Interpolated normal (then normalize)
                nrm = u * v_nrm[:, 0] + v * v_nrm[:, 1] + w * v_nrm[:, 2]
                nrm = nrm / (np.linalg.norm(nrm) + 1e-8)

                # Interpolated UV
                uv = u * v_uvs[0] + v * v_uvs[1] + w * v_uvs[2]
                tu = np.clip(int(uv[0] * (tex.shape[1] - 1)), 0, tex.shape[1] - 1)
                tv = np.clip(int(uv[1] * (tex.shape[0] - 1)), 0, tex.shape[0] - 1)
                tv = np.clip(int((1 - uv[1]) * (tex.shape[0] - 1)), 0, tex.shape[0] - 1)

                
                vclr = tex[tv, tu, :]   # Normalize to [0,1]
              

                # Interpolated 3D position (optional, but passed for consistency)
                pt = u * v_pos[:, 0] + v * v_pos[:, 1] + w * v_pos[:, 2]

                # Lighting calculation using fixed V and L
                color = light(pt, nrm, vclr, cam_pos, mat, l_pos, l_int, l_amb, fixed_V=V, fixed_L_list=L_list)
                img[j, i, :] = np.clip(color, 0, 1)

    return img




# # Example usage of the shade_phong function (uncomment the following lines to test):

# # ----------------------------------------------------------------------------
# # DEMO 1: Phong Shading Test 
# # ----------------------------------------------------------------------------

# import matplotlib.pyplot as plt
# from shade_gouraud_func import shade_gouraud

# # Create a solid red texture
# tex = np.ones((64, 64, 3), dtype=np.uint8) * [255, 0, 0]

# # Output image 
# res_h, res_w = 100, 100
# img = np.zeros((res_h, res_w, 3), dtype=np.float32)

# # Create two output buffers
# img_phong = np.zeros((res_h, res_w, 3), dtype=np.float32)

# # Triangle screen-space projection
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

# # Render both Gouraud and Phong
# img_phong = shade_phong(v_pos, v_nrm, v_uvs, tex, cam_pos, mat, l_pos, l_int, l_amb, img_phong)

# plt.imshow(img_phong)
# plt.title("Phong Shading")
# plt.axis('off')

# plt.show()

