from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from vector_interp import vector_interp


@dataclass
class Edge:
    """Represents an edge of a triangle with texture coordinates."""

    min_y: int
    max_y: int
    min_x: float
    max_x: float
    m: float
    p1: Tuple[int, int]
    uv1: np.ndarray
    p2: Tuple[int, int]
    uv2: np.ndarray

    def __init__(
        self, p1: List[int], p2: List[int], uv1: np.ndarray, uv2: np.ndarray
    ):
        x1, y1 = p1
        x2, y2 = p2

        self.p1 = p1
        self.p2 = p2
        self.uv1 = uv1
        self.uv2 = uv2

        self.min_y = min(y1, y2)
        self.max_y = max(y1, y2)
        self.min_x = x1 if y1 == self.min_y else x2
        self.max_x = x1 if y1 == self.max_y else x2

        dx = x2 - x1
        dy = y2 - y1
        self.m = dy / dx if dx != 0 else float("inf")


def t_shading(
    img: np.ndarray,
    vertices: List[List[int]],
    uv: np.ndarray,
    textImg: np.ndarray,
) -> np.ndarray:
    height, width = img.shape[:2]
    tex_h, tex_w = textImg.shape[:2]

    # Build edges from triangle vertices
    edges = [
        Edge(vertices[i], vertices[(i + 1) % 3], uv[i], uv[(i + 1) % 3])
        for i in range(3)
    ]

    ys = [v[1] for v in vertices]
    min_y_total = max(0, min(ys))
    max_y_total = min(height - 1, max(ys))

    for y in range(min_y_total, max_y_total + 1):
        # Get active edges for this scanline
        active_edges = [e for e in edges if e.min_y <= y < e.max_y]
        if len(active_edges) != 2:
            continue  # Skip if not exactly two intersections

        x_intersects = []
        uv_intersects = []

        for edge in active_edges:
            # Interpolate x
            dy = edge.p2[1] - edge.p1[1]
            if dy != 0:
                t = (y - edge.p1[1]) / dy
                x = edge.p1[0] + t * (edge.p2[0] - edge.p1[0])
            else:
                x = edge.p1[0]  # Horizontal edge case

            uv_interp = vector_interp(
                edge.p1, edge.p2, edge.uv1, edge.uv2, y, dim=2
            )

            x_intersects.append(x)
            uv_intersects.append(uv_interp)

        # Sort intersections from left to right
        if x_intersects[0] < x_intersects[1]:
            x0, x1 = x_intersects
            uv0, uv1 = uv_intersects
        else:
            x0, x1 = x_intersects[::-1]
            uv0, uv1 = uv_intersects[::-1]

        x_start = max(0, int(np.ceil(x0)))
        x_end = min(width - 1, int(np.floor(x1)))

        for x in range(x_start, x_end + 1):
            uv_interp = vector_interp((x0, y), (x1, y), uv0, uv1, x, dim=1)

            # Convert UV to texture space
            u = np.clip(uv_interp[0], 0.0, 1.0)
            v = np.clip(uv_interp[1], 0.0, 1.0)

            u_pixel = int(u * (tex_w - 1))
            v_pixel = int(v * (tex_h - 1))

            color = textImg[v_pixel, u_pixel]
            img[y, x] = color / 255.0

    return img
