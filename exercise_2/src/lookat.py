from typing import Tuple
import numpy as np

def lookat(eye: np.ndarray, up: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the camera's view matrix (i.e., its coordinate frame transformation
    specified by a rotation matrix R, and a translation vector t).
    
    :return: a tuple containing:
        - R: rotation matrix (3 x 3)
        - t: translation vector (1 x 3)
    """
    zc = target - eye
    zc /= np.linalg.norm(zc)

    yc = up - np.dot(up, zc) * zc
    yc /= np.linalg.norm(yc)

    xc = np.cross(yc, zc)

    R = np.stack((xc, yc, zc), axis=1)  # Columns: x_c, y_c, z_c
    t = eye.reshape(1, 3)

    return R, t
