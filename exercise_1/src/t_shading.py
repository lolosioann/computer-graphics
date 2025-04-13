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
    """
    Applies texture mapping to a triangle defined by 2D vertices and
    per-vertex UV coordinates.

    Parameters
    ----------
    img : np.ndarray
        Image buffer (H x W x 3), float [0, 1].
    vertices : List[List[int]]
        List of three [x, y] pairs representing triangle vertices.
    uv : np.ndarray
        (3 x 2) array of [u, v] texture coordinates for each vertex.
    textImg : np.ndarray
        Texture image (H x W x 3), uint8 in [0, 255].

    Returns
    -------
    np.ndarray
        Image with textured triangle drawn.
    """
    height, width = img.shape[:2]
    tex_h, tex_w = textImg.shape[:2]

    # Build edges
    edges = [
        Edge(vertices[i], vertices[(i + 1) % 3], uv[i], uv[(i + 1) % 3])
        for i in range(3)
    ]

    # Triangle scanline bounds
    ys = [v[1] for v in vertices]
    min_y_total, max_y_total = max(0, min(ys)), min(height - 1, max(ys))

    active_edges: List[Edge] = []

    # Scanline algorithm
    for y in range(min_y_total, max_y_total + 1):
        # activate/deactivate edges
        for edge in edges:
            if edge.min_y == y and edge.m != 0:
                active_edges.append(edge)
            elif edge.max_y == y and edge in active_edges:
                active_edges.remove(edge)
    
        # add active points based on edges
        x_intersects = []
        uv_intersects = []
        for edge in active_edges:
            if edge.m == float("inf"):
                x_intersect = edge.min_x
            else:
                if edge.m > 0:
                    x_intersect = edge.min_x + (y - edge.min_y) / edge.m
                else:
                    x_intersect = edge.max_x - (y - edge.max_y) / edge.m
            x_intersects.append(x_intersect)
            uv_interp = vector_interp(
                edge.p1, edge.p2, edge.uv1, edge.uv2, y, dim=2
            )
            uv_intersects.append(uv_interp)


            # if there are two active points (because we have triangles), fill
            # the pixels
        if len(x_intersects) == 2:
            if x_intersects[0] < x_intersects[1]:
                x0, x1 = x_intersects
                uv0, uv1 = uv_intersects
            else:
                x0, x1 = x_intersects[::-1]
                uv0, uv1 = uv_intersects[::-1]

            x_start = max(0, int(np.ceil(x0)))
            x_end = min(width - 1, int(np.floor(x1)))

            for x in range(x_start, x_end):
                uv_interp = vector_interp((x0, y), (x1, y), uv0, uv1, x, dim=1)

                # Convert to texture pixel coordinates
                u_pixel = int(
                    np.clip(uv_interp[0] * (tex_w - 1), 0, tex_w - 1)
                )
                v_pixel = int(
                    np.clip(uv_interp[1] * (tex_h - 1), 0, tex_h - 1)
                )

                color = textImg[v_pixel, u_pixel]
                img[y, x] = color / 255.0

    return img


# for y in range(min_y_total, max_y_total + 1):
#         x_intersections = []
#         uv_intersections = []

#         for edge in edges:
#             if edge.ymin <= y <= edge.ymax:
#                 # Interpolate x
#                 if edge.p1[1] < edge.p2[1]:
#                     x = edge.p1[0] + (y - edge.p1[1]) * edge.inv_slope
#                     uv_interp = vector_interp(
#                         edge.p1, edge.p2, edge.uv1, edge.uv2, y, dim=2
#                     )
#                 else:
#                     x = edge.p2[0] + (y - edge.p2[1]) * edge.inv_slope
#                     uv_interp = vector_interp(
#                         edge.p2, edge.p1, edge.uv2, edge.uv1, y, dim=2
#                     )

#                 x_intersections.append(x)
#                 uv_intersections.append(uv_interp)