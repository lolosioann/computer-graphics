from typing import Tuple
import numpy as np

def lookat(eye: np.ndarray, up: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the camera coordinate frame (rotation matrix R and translation vector t)
    that aligns the camera to look from `eye` towards `target`, with the given `up` vector.
    
    The resulting frame (x_c, y_c, z_c) follows the standard right-handed convention:
        - z_c: forward direction (from eye to target)
        - x_c: right direction (perpendicular to up and z_c)
        - y_c: camera up direction

    Parameters:
        eye (np.ndarray): The camera position in world coordinates (3,).
        up (np.ndarray): The up vector in world coordinates (3,).
        target (np.ndarray): The point the camera is looking at (3,).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - R (3x3 ndarray): Camera-to-world rotation matrix.
            - t (1x3 ndarray): Camera origin (eye position) as a row vector.
    """
    # Normalize forward (view direction)
    z_c = target - eye
    z_c /= np.linalg.norm(z_c)

    # Make up orthogonal to forward
    y_c = up - np.dot(up, z_c) * z_c
    y_c /= np.linalg.norm(y_c)

    x_c = np.cross(y_c, z_c)

    # Assemble rotation matrix
    R = np.stack((x_c, y_c, z_c), axis=1)

    # t vector: camera origin in world coordinates
    t = eye.reshape(1, 3)

    return R, t
