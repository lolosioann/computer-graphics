from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Edge:
    """Represents an edge of a triangle with slope and bounding information."""

    min_y: int
    max_y: int
    min_x: float
    max_x: float
    m: float

    def __init__(self, p1: List[int], p2: List[int]):
        x1, y1 = p1
        x2, y2 = p2

        self.min_y = min(y1, y2)
        self.max_y = max(y1, y2)
        self.min_x = x1 if y1 == self.min_y else x2
        self.max_x = x1 if y1 == self.max_y else x2

        dx = x2 - x1
        dy = y2 - y1
        self.m = dy / dx if dx != 0 else float("inf")

    # For debugging purposes
    def __repr__(self):
        return (
            f"Edge(min_y={self.min_y}, max_y={self.max_y}, "
            f"min_x={self.min_x}, max_x={self.max_x}, m={self.m})"
        )


def f_shading(
    img: np.ndarray, vertices: List[List[int]], vcolors: List[List[float]]
) -> np.ndarray:
    """
    Applies flat shading to a triangle defined by three 2D vertices and
    their associated colors.

    Parameters
    ----------
    img : np.ndarray
        Image buffer (H x W x 3), float [0, 1].
    vertices : List[List[int]]
        List of three [x, y] pairs representing triangle vertices.
    vcolors : List[List[float]]
        List of three [r, g, b] colors in [0, 1] range for each vertex.

    Returns
    -------
    np.ndarray
        The image with the triangle shaded using the average color.
    """
    height, width = img.shape[:2]
    avg_color = np.mean(vcolors, axis=0)

    # Construct triangle edges
    edges = [Edge(vertices[i], vertices[(i + 1) % 3]) for i in range(3)]

    # Determine the bounding box of the triangle (min and max y)
    ys = [v[1] for v in vertices]
    min_y_total = max(0, min(ys))
    max_y_total = min(height - 1, max(ys))

    for y in range(min_y_total, max_y_total + 1):
        # Get all edges active at this scanline
        active_edges = [e for e in edges if e.min_y <= y < e.max_y]

        x_intersects = []
        for edge in active_edges:
            if edge.m == float("inf"):
                x_intersect = edge.min_x
            else:
                x_intersect = edge.min_x + (y - edge.min_y) / edge.m
            x_intersects.append(x_intersect)

        if len(x_intersects) == 2:
            x_start = max(0, int(np.floor(min(x_intersects))))
            x_end = min(width - 1, int(np.ceil(max(x_intersects))))
            img[y, x_start:x_end] = avg_color

    return img
