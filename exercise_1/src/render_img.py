import numpy as np

from f_shading import f_shading
from t_shading import t_shading


def render_img(
    faces, vertices, vcolors, uvs, depth, shading, texImg
):
    """
    Render an image with either flat or texture shading.

    Parameters:
    - faces: Kx3 array of vertex indices forming triangles.
    - vertices: Lx2 array of 2D vertex positions.
    - vcolors: Lx3 array of vertex RGB colors.
    - uvs: Lx2 array of texture coordinates.
    - depth: L array of depth values.
    - shading: 'f' for flat shading, 't' for texture shading.
    - texImg: texture image for 't' shading.

    Returns:
    - img: 512x512x3 RGB image (float [0, 1]).
    """
    M, N = 512, 512
    img = np.ones((M, N, 3))  # white background

    # Store (average_depth, face) tuples
    face_depths = []
    for face in faces:
        avg_depth = np.mean(depth[face])
        face_depths.append((avg_depth, face))

    # Sort by average depth (back to front)
    face_depths.sort(key=lambda x: x[0], reverse=True)

    # Render each triangle
    for avg_depth, face in face_depths:
        idxs = face
        tri_vertices = vertices[idxs]
        tri_colors = vcolors[idxs]        
        tri_uvs = uvs[idxs]               

        if shading == 'f':
            img = f_shading(img, tri_vertices, tri_colors)
        elif shading == 't':
            img = t_shading(img, tri_vertices, tri_uvs, texImg)
        else:
            raise ValueError("Shading must be 'f' or 't'.")

    return img
