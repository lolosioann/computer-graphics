import numpy as np

from f_shading import f_shading
from t_shading import t_shading


def render_img(
    faces, vertices, vcolors, depth, shading, text_img=None, uvs=None
):
    """
    Render an image with either flat or texture shading.

    Parameters:
    - faces: K×3 array of vertex indices forming triangles.
    - vertices: L×2 array of 2D vertex positions.
    - vcolors: L×3 array of vertex RGB colors.
    - depth: L×1 array of depth values.
    - shading: 'f' for flat shading, 't' for texture shading.
    - text_img: optional texture image for 't' shading.

    Returns:
    - img: 512×512×3 RGB image.
    """
    M, N = 512, 512
    img = np.ones((M, N, 3))  # white background

    # Compute triangle depth (mean of vertex depths)
    triangle_depths = np.mean(depth[faces], axis=1).flatten()

    # Sort triangles from farthest to closest
    sorted_indices = np.argsort(-triangle_depths)
    faces_sorted = faces[sorted_indices]

    for face in faces_sorted:
        tri_vertices = vertices[face]
        if shading == "f":
            tri_colors = vcolors[face]
            img = f_shading(img, tri_vertices, tri_colors)
        elif shading == "t":
            tri_uvs = uvs[face]
            img = t_shading(img, tri_vertices, tri_uvs, text_img)
        else:
            raise ValueError("Invalid shading mode. Use 'f' or 't'.")

    return img
