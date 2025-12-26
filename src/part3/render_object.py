from typing import List, Union

import numpy as np

from calc_normals import calc_normals
from lookat import lookat
from MatPhong import MatPhong
from perspective_project import perspective_project
from rasterize import rasterize
from shade_gouraud import shade_gouraud
from shade_phong import shade_phong

def render_object(
    v_pos: np.ndarray,
    v_uvs: np.ndarray,
    t_pos_idx: np.ndarray,
    tex: np.ndarray,
    plane_h: int,
    plane_w: int,
    res_h: int,
    res_w: int,
    focal: float,
    eye: np.ndarray,
    up: np.ndarray,
    target: np.ndarray,
    mat: MatPhong,
    l_pos: Union[np.ndarray, List[np.ndarray]],
    l_int: Union[np.ndarray, List[np.ndarray]],
    l_amb: np.ndarray,
    shader: str
) -> np.ndarray:
    """
    This function renders a textured 3D object onto a 2D image using either Gouraud or Phong shading. It:    

    1. Computes vertex normals using face connectivity.
    2. Applies camera transformation using the LookAt model.
    3. Projects 3D vertices onto a 2D image plane with perspective projection.
    4. Rasterizes the 2D coordinates into pixel space.
    5. Loops through each triangle in the mesh to render it individually using the selected shading model.
    
    Parameters:
    - v_pos: (3, Nv) array of 3D vertex positions.
    - v_uvs: (Nv, 2) array of texture coordinates (u, v) per vertex.
    - t_pos_idx: (3, Nt) array of triangle indices (0-based).
    - tex: (H, W, 3) texture image.
    - plane_h, plane_w: Physical size of the image plane (in scene units).
    - res_h, res_w: Resolution of the output image in pixels.
    - focal: Focal length of the virtual camera.
    - eye, up, target: Camera setup parameters.
    - mat: Instance of MatPhong defining surface material.
    - l_pos, l_int: Light positions and their intensities.
    - l_amb: Ambient light intensity.
    - shader: Either 'gouraud' or 'phong' to select the shading model.
    
    Returns:
    - img: (res_h, res_w, 3) float image with RGB values in [0, 1].
    """

    # Step 1: Calculate vertex normals
    v_normals = calc_normals(v_pos, t_pos_idx)

    # Step 2: LookAt transformation
    R, t = lookat(eye, up, target)

    # Step 3: Perspective projection
    # perspective_project expects (N, 3) but v_pos is (3, N), so transpose
    proj_pts, depth = perspective_project(v_pos.T, focal, R, t)  

    # Step 4: Rasterize to screen coordinates
    screen_pts = rasterize(proj_pts, plane_w, plane_h, res_w, res_h)  # (N, 2)

    # Step 5: Combine with depth
    # Convert screen_pts from (N, 2) to (2, N) and add depth as 3rd row
    screen_pts_T = screen_pts.T  # (2, N)
    depth_row = depth.reshape(1, -1)  # (1, N)
    vertices_2d = np.vstack([screen_pts_T, depth_row])  # (3, N)  

    # Step 6: Initialize image
    img = np.ones((res_h, res_w, 3), dtype=np.float32)

    # Step 7: Loop over triangles
    for triangle_idx in range(t_pos_idx.shape[1]):
        idx = t_pos_idx[:, triangle_idx]  

        # Skip if any index is invalid
        if np.any(idx >= v_pos.shape[1]) or np.any(idx < 0):
            continue

        # Get triangle data
        tri_v_pos = vertices_2d[:, idx]      
        tri_v_nrm = v_normals[:, idx]        
        tri_v_uvs = v_uvs[idx, :]            

        # Choose shading mode
        if shader == 'gouraud':
            img = shade_gouraud(
                v_pos=tri_v_pos,
                v_nrm=tri_v_nrm,
                v_uvs=tri_v_uvs,
                tex=tex,
                cam_pos=eye,
                mat=mat,
                l_pos=l_pos,
                l_int=l_int,
                l_amb=l_amb,
                img=img
            )
        elif shader == 'phong':
            img = shade_phong(
                v_pos=tri_v_pos,
                v_nrm=tri_v_nrm,
                v_uvs=tri_v_uvs,
                tex=tex,
                cam_pos=eye,
                mat=mat,
                l_pos=l_pos,
                l_int=l_int,
                l_amb=l_amb,
                img=img
            )
        else:
            raise ValueError(f"Unknown shader type: {shader}")

    return img
