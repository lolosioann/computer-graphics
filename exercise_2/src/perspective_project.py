from typing import Tuple
import numpy as np
from world2view import world2view

def perspective_project(
    pts: np.ndarray,
    focal: float,
    R: np.ndarray,
    t: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D world points onto a 2D image plane using a pinhole camera model.
    
    Parameters:
    - pts: (N, 3) array of 3D points in world coordinates.
    - focal: scalar focal length.
    - R: (3, 3) rotation matrix.
    - t: (3,) translation vector (camera center in world coordinates).
    
    Returns:
    - proj_pts: (N, 2) projected 2D points.
    - depths: (N,) depth values (Z in the camera frame).
    """
    # Step 1: Transform to camera coordinate frame using your function
    cam_pts = world2view(pts, R, t)  # shape (N, 3)

    # Step 2: Perspective division using focal length
    X = cam_pts[:, 0]
    Y = cam_pts[:, 1]
    Z = cam_pts[:, 2]

    # Avoid divide-by-zero errors
    if np.any(Z == 0):
        raise ZeroDivisionError("Some points lie on the camera plane (Z=0)")

    proj_x = focal * X / Z
    proj_y = focal * Y / Z
    proj_pts = np.stack([proj_x, proj_y], axis=1)  # (N, 2)

    return proj_pts, Z
