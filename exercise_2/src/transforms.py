import numpy as np

def translate(t_vec: np.ndarray) -> np.ndarray:
    """
    Create an affine transformation matrix w.r.t. the
    specified translation vector.
    """
    xform = np.eye(4)
    xform[0:3, 3] = t_vec

    return xform

import numpy as np

def rotate(axis: np.ndarray, angle: float, center: np.ndarray = np.zeros(3)) -> np.ndarray:
    """
    Create a 4x4 affine transformation matrix representing a rotation around
    a specified axis by a given angle and about a specific center point.

    Parameters:
    - axis (np.ndarray): A 3-element array specifying the rotation axis.
    - angle (float): The rotation angle in radians.
    - center (np.ndarray): A 3-element point about which to perform the rotation.

    Returns:
    - np.ndarray: A 4x4 affine transformation matrix.
    """


    # Normalize axis
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("Rotation axis cannot be the zero vector.")
    ux, uy, uz = axis / norm

    c = np.cos(angle)
    s = np.sin(angle)
    one_c = 1 - c

    # Rodrigues' rotation formula components
    R = np.array([
        [c + ux**2 * one_c,      ux*uy*one_c - uz*s,     ux*uz*one_c + uy*s],
        [uy*ux*one_c + uz*s,     c + uy**2 * one_c,      uy*uz*one_c - ux*s],
        [uz*ux*one_c - uy*s,     uz*uy*one_c + ux*s,     c + uz**2 * one_c ]
    ], dtype=np.float64)

    # Embed in 4x4 affine matrix
    xform = np.eye(4, dtype=np.float64)
    xform[:3, :3] = R

    # Adjust for center of rotation
    translation = center - R @ center
    xform[:3, 3] = translation

    return xform


def compose(mat1: np.ndarray, mat2: np.ndarray) -> np.ndarray:
    """
    Compose two 4x4 affine transformation matrices using matrix multiplication.

    The result is equivalent to applying `mat1` first, then `mat2`.

    Parameters:
    - mat1 (np.ndarray): First affine transformation matrix (applied first).
    - mat2 (np.ndarray): Second affine transformation matrix (applied second).

    Returns:
    - np.ndarray: The composed 4x4 affine transformation matrix.
    """
    if mat1.shape != (4, 4) or mat2.shape != (4, 4):
        raise ValueError("Both input matrices must be 4x4 affine transformation matrices.")

    mat = np.matmul(mat2, mat1)  
    return mat

# This one is a helper function
def convert2affine(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Convert a rotation matrix and translation vector into a 4x4 affine transformation matrix.

    Parameters:
    - R (np.ndarray): A 3x3 rotation matrix.
    - t (np.ndarray): A 3-element translation vector.

    Returns:
    - np.ndarray: A 4x4 affine transformation matrix.
    """
    if R.shape != (3, 3):
        raise ValueError("Rotation matrix must be 3x3.")
    if t.shape != (3,):
        raise ValueError("Translation vector must be 3 elements.")

    affine = np.eye(4)
    affine[:3, :3] = R
    affine[:3, 3] = t

    return affine
