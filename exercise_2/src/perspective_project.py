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
    Projects 3D points from world space onto a 2D image plane using a pinhole camera model.

    Parameters:
        pts (np.ndarray): Array of shape (N, 3), representing 3D points in world coordinates.
        focal (float): Focal length (assumed same in both x and y directions).
        R (np.ndarray): Rotation matrix of shape (3, 3), defining camera orientation.
        t (np.ndarray): Translation vector of shape (3,), defining camera center in world space.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - proj_pts: (N, 2) array of projected 2D image coordinates.
            - depths: (N,) array of depth values in the camera coordinate system.
    """
    # Transform points from world space to camera space
    cam_pts = world2view(pts, R, t)  # (N, 3)

    X, Y, Z = cam_pts[:, 0], cam_pts[:, 1], cam_pts[:, 2]

    # Guard for divide-by-zero 
    if np.any(np.isclose(Z, 0)):
        raise ZeroDivisionError("One or more points project to infinity (Z ≈ 0 in camera space)")

    # Perspective division
    proj_x = focal * X / Z
    proj_y = focal * Y / Z
    proj_pts = np.stack((proj_x, proj_y), axis=1)  # (N, 2)

    return proj_pts, Z
