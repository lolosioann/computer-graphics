import numpy as np
from typing import Union

def translate(t_vec: np.ndarray) -> np.ndarray:
    """
    Generate a 4x4 affine transformation matrix representing a translation.

    Parameters:
    - t_vec (np.ndarray): A 3-element vector specifying the translation.

    Returns:
    - np.ndarray: A 4x4 affine transformation matrix applying the translation.
    """
    if t_vec.shape != (3,):
        raise ValueError("Translation vector must be a 3-element vector.")

    xform = np.eye(4, dtype=np.float64)
    xform[:3, 3] = t_vec

    return xform


def rotate(axis: np.ndarray, angle: float, center: np.ndarray = np.zeros(3)) -> np.ndarray:
    """
    Generate a 4x4 affine transformation matrix for rotation around an arbitrary axis.

    Parameters:
    - axis (np.ndarray): A 3-element array defining the rotation axis.
    - angle (float): Rotation angle in radians.
    - center (np.ndarray): A 3-element point about which to rotate.

    Returns:
    - np.ndarray: A 4x4 affine matrix representing the rotation.
    """
    if axis.shape != (3,):
        raise ValueError("Rotation axis must be a 3-element vector.")
    if center.shape != (3,):
        raise ValueError("Center of rotation must be a 3-element vector.")

    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("Rotation axis cannot be the zero vector.")

    ux, uy, uz = axis / norm
    c = np.cos(angle)
    s = np.sin(angle)
    one_c = 1 - c

    # Rodrigues' formula
    R = np.array([
        [c + ux**2 * one_c,    ux*uy * one_c - uz*s,  ux*uz * one_c + uy*s],
        [uy*ux * one_c + uz*s, c + uy**2 * one_c,     uy*uz * one_c - ux*s],
        [uz*ux * one_c - uy*s, uz*uy * one_c + ux*s,  c + uz**2 * one_c   ]
    ], dtype=np.float64)

    # Build affine matrix
    xform = np.eye(4, dtype=np.float64)
    xform[:3, :3] = R
    xform[:3, 3] = center - R @ center  # Translation to maintain center of rotation

    return xform


def compose(mat1: np.ndarray, mat2: np.ndarray) -> np.ndarray:
    """
    Compose two 4x4 affine transformations via matrix multiplication.

    The result corresponds to applying `mat1` first, then `mat2`.

    Parameters:
    - mat1 (np.ndarray): First transformation (4x4).
    - mat2 (np.ndarray): Second transformation (4x4).

    Returns:
    - np.ndarray: Composed transformation (4x4).
    """
    if mat1.shape != (4, 4) or mat2.shape != (4, 4):
        raise ValueError("Both input matrices must be 4x4 affine matrices.")

    return mat2 @ mat1


def convert2affine(R: np.ndarray, t: Union[np.ndarray, list, tuple]) -> np.ndarray:
    """
    Convert rotation matrix and translation vector into a 4x4 affine transformation.

    Parameters:
    - R (np.ndarray): A 3x3 rotation matrix.
    - t (np.ndarray | list | tuple): A translation vector of shape (3,).

    Returns:
    - np.ndarray: A 4x4 affine transformation matrix.
    """
    t = np.asarray(t, dtype=np.float64)

    if R.shape != (3, 3):
        raise ValueError("Rotation matrix must be 3x3.")
    if t.shape != (3,):
        raise ValueError("Translation vector must be a 3-element vector.")

    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = R
    affine[:3, 3] = t

    return affine
