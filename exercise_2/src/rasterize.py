import numpy as np

def rasterize(
    pts_2d: np.ndarray,
    plane_w: int,
    plane_h: int,
    res_w: int,
    res_h: int
) -> np.ndarray:
    
    x = pts_2d[:, 0]
    y = pts_2d[:, 1]

    # Normalized coordinates [0,1]
    u = (x + plane_w / 2) / plane_w
    v = (y + plane_h / 2) / plane_h

    # Map to pixel coordinates
    col = np.floor(u * res_w).astype(int)
    row = np.floor((1 - v) * res_h).astype(int)  # y-axis inverted

    # Clamp values inside image bounds
    col = np.clip(col, 0, res_w - 1)
    row = np.clip(row, 0, res_h - 1)

    return np.stack([col, row], axis=1)
