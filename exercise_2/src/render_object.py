import numpy as np
from perspective_project import perspective_project
from lookat import lookat
from rasterize import rasterize
from render_img import render_img

def render_object(
    v_pos: np.ndarray,
    v_clr: np.ndarray,
    t_pos_idx: np.ndarray,
    plane_h: int,
    plane_w: int,
    res_h: int,
    res_w: int,
    focal: float,
    eye: np.ndarray,
    up: np.ndarray,
    target: np.ndarray,
    uvs: np.ndarray = None,
    texImg: np.ndarray = None
) -> np.ndarray:
    """
    Render a 3D object from a given camera configuration.

    Parameters:
    - v_pos: (N, 3) array of vertex positions in world space
    - v_clr: (N, 3) array of vertex colors (RGB)
    - t_pos_idx: (M, 3) array of triangle vertex indices
    - plane_h, plane_w: physical dimensions of the camera plane
    - res_h, res_w: image resolution in pixels
    - focal: focal length of the camera
    - eye: (3,) camera origin
    - up: (3,) up direction vector
    - target: (3,) target point the camera looks at

    Returns:
    - (res_h, res_w, 3) RGB image rendered from the specified view
    """ 

    # Compute camera rotation and translation
    R, t = lookat(eye, up, target)

    xy_coords, depths = perspective_project(v_pos, focal, R, t)
    v_pix = rasterize(xy_coords, plane_w, plane_h, res_w, res_h)
    # uvs = np.ones((v_pix.shape[0], 2))  # Dummy UVs with correct shape

    if uvs is None or texImg is None:
        img = render_img(t_pos_idx, v_pix, v_clr, uvs=uvs, depth=depths, shading="f", texImg=None, M=res_h, N=res_w)
    else:
        img = render_img(t_pos_idx, v_pix, v_clr, uvs=uvs, depth=depths, shading="t", texImg=texImg, M=res_h, N=res_w)

    return img