from typing import Tuple, Union

import numpy as np


def vector_interp(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    V1: Union[float, Tuple[float, ...]],
    V2: Union[float, Tuple[float, ...]],
    coord: float,
    dim: int,
) -> np.ndarray:
    """
    Performs linear interpolation between V1 and V2 based on a coordinate value
    along a specified axis (1 for x, 2 for y) between two points p1 and p2.

    Parameters
    ----------
    p1 : Tuple[float, float]
        Coordinates of the first point.
    p2 : Tuple[float, float]
        Coordinates of the second point.
    V1 : Union[float, Tuple[float, ...]]
        Value or vector associated with p1.
    V2 : Union[float, Tuple[float, ...]]
        Value or vector associated with p2.
    coord : float
        Coordinate at which to interpolate.
    dim : int
        Dimension to interpolate along: 1 for x, 2 for y.

    Returns
    -------
    np.ndarray
        Interpolated value as a NumPy array.
    """
    V1 = np.asarray(V1, dtype=float)
    V2 = np.asarray(V2, dtype=float)

    if np.array_equal(p1, p2):
        return V1.copy()

    if dim not in {1, 2}:
        raise ValueError("dim must be 1 (x) or 2 (y)")

    # Select dimension index (0 for x, 1 for y)
    idx = dim - 1
    p1_dim, p2_dim = p1[idx], p2[idx]

    # Check bounds
    if not (min(p1_dim, p2_dim) <= coord <= max(p1_dim, p2_dim)):
        raise ValueError(
            "coord is out of bounds for the "
            f"{'x' if dim == 1 else 'y'} dimension"
        )

    # Avoid division by zero
    if p1_dim == p2_dim:
        return V1.copy()

    # Linear interpolation factor
    alpha = (coord - p1_dim) / (p2_dim - p1_dim)
    return (1 - alpha) * V1 + alpha * V2
