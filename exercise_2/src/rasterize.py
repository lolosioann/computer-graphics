import numpy as np

def rasterize(
    pts_2d: np.ndarray,
    plane_w: float,
    plane_h: float,
    res_w: int,
    res_h: int
) -> np.ndarray:
    """
    Convert 2D points from image plane space to pixel (raster) coordinates.

    Parameters:
        pts_2d (np.ndarray): Array of shape (N, 2), where each row is a 2D point (x, y)
                             in the image (camera) plane coordinates.
        plane_w (float): Width of the image (sensor) plane in world units.
        plane_h (float): Height of the image (sensor) plane in world units.
        res_w (int): Horizontal resolution (number of pixels along width).
        res_h (int): Vertical resolution (number of pixels along height).

    Returns:
        np.ndarray: Array of shape (N, 2) with pixel coordinates (col, row).
    """

    # Extract x and y coordinates
    x = pts_2d[:, 0]
    y = pts_2d[:, 1]

    # Normalize to range [0, 1]
    u = (x + plane_w / 2) / plane_w
    v = (y + plane_h / 2) / plane_h

    # Convert to pixel coordinates
    col = np.floor(u * res_w).astype(int)
    row = np.floor((1 - v) * res_h).astype(int)  # y-axis inverted for image coordinates

    # Clamp to valid pixel range
    col = np.clip(col, 0, res_w - 1)
    row = np.clip(row, 0, res_h - 1)

    return np.stack((col, row), axis=1)
