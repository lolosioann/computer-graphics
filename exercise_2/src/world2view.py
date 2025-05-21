import numpy as np
from transforms import convert2affine

def world2view(pts: np.ndarray, R: np.ndarray, c0: np.ndarray) -> np.ndarray:
    """
    Transforms points from the world coordinate frame to the camera (view) frame.

    Parameters:
    - pts: (N, 3) array of 3D points in world coordinates
    - R: (3, 3) rotation matrix of the camera w.r.t. the world
    - c0: (3,) camera center in world coordinates

    Returns:
    - (N, 3) array of 3D points in camera coordinates
    """
    T = convert2affine(R.T, -np.matmul(R.T, c0))  # world-to-view affine matrix
    pts_homogeneous = np.hstack((pts, np.ones((pts.shape[0], 1))))
    return np.matmul(T, pts_homogeneous.T).T[:, :3]
