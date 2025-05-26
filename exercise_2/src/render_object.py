import numpy as np
from perspective_project import perspective_project
from lookat import lookat
from rasterize import rasterize
from render_img import render_img

def render_object(
    v_pos: np.ndarray,
    v_clr: np.ndarray,
    t_pos_idx: np.ndarray,
    plane_h: float,
    plane_w: float,
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
    Render a 3D object from a specified camera configuration using a basic rasterization pipeline.

    Parameters:
        v_pos (np.ndarray): Vertex positions in world space, shape (N, 3).
        v_clr (np.ndarray): Vertex colors (RGB), shape (N, 3), values in [0, 1].
        t_pos_idx (np.ndarray): Triangle indices, shape (M, 3), referring to v_pos/v_clr rows.
        plane_h (float): Sensor (image plane) height in world units.
        plane_w (float): Sensor (image plane) width in world units.
        res_h (int): Image height in pixels.
        res_w (int): Image width in pixels.
        focal (float): Focal length of the pinhole camera.
        eye (np.ndarray): Camera position in world coordinates, shape (3,).
        up (np.ndarray): Camera up direction, shape (3,).
        target (np.ndarray): Look-at target point, shape (3,).
        uvs (np.ndarray, optional): Vertex UV coordinates, shape (N, 2).
        texImg (np.ndarray, optional): Texture image, shape (H, W, 3), with values in [0, 1].

    Returns:
        np.ndarray: Rendered image as an array of shape (res_h, res_w, 3), with float values in [0, 1].
    """

    # Compute camera rotation and translation matrix
    R, t = lookat(eye, up, target)

    # Project 3D vertices into 2D camera coordinates
    xy_coords, depths = perspective_project(v_pos, focal, R, t)

    # Convert to pixel coordinates
    v_pix = rasterize(xy_coords, plane_w, plane_h, res_w, res_h)

    # Render final image 
    if uvs is None or texImg is None:
        # Flat shading
        img = render_img(
            t_pos_idx, v_pix, v_clr,
            uvs=None, depth=depths,
            shading="f", texImg=None,
            M=res_h, N=res_w
        )
    else:
        # render with texture shading
        img = render_img(
            t_pos_idx, v_pix, v_clr,
            uvs=uvs, depth=depths,
            shading="t", texImg=texImg,
            M=res_h, N=res_w
        )

    return img
