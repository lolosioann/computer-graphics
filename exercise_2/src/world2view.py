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
    c0 = c0.reshape(-1)  # Ensure it’s a 1D vector of shape (3,)
    
    # Construct the world-to-camera affine transformation matrix
    T = convert2affine(R.T, -np.matmul(R.T, c0))
    
    # Convert points to homogeneous coordinates (N x 4)
    pts_homogeneous = np.hstack((pts, np.ones((pts.shape[0], 1))))
    
    # Apply transformation and drop homogeneous coordinate
    return np.matmul(T, pts_homogeneous.T).T[:, :3]
