import numpy as np

def calc_normals(pts: np.ndarray, t_pos_idx: np.ndarray) -> np.ndarray:
    """
    Calculate vertex normals from triangle mesh.

    Parameters:
    - pts: 3 x Nv array of vertex positions (float64).
    - t_pos_idx: 3 x Nt array of triangle indices (1-based indexing).

    Returns:
    - 3 x Nv array of normalized vertex normals.
    """
    # Ensure input is float64 to avoid dtype casting errors
    pts = pts.astype(np.float64)

    Nv = pts.shape[1]
    Nt = t_pos_idx.shape[1]

    # Initialize normal vectors with zeros
    normals = np.zeros((3, Nv), dtype=np.float64)

    for k in range(Nt):
        # Convert from 1-based to 0-based indexing
        i0, i1, i2 = t_pos_idx[:, k] 

        v0 = pts[:, i0]
        v1 = pts[:, i1]
        v2 = pts[:, i2]

        # Compute the face normal
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normal = np.cross(edge1, edge2).astype(np.float64)

        # Normalize the face normal
        norm = np.linalg.norm(face_normal)
        if norm > 1e-8:
            face_normal /= norm

        # Add the face normal to each vertex of the triangle
        normals[:, i0] += face_normal
        normals[:, i1] += face_normal
        normals[:, i2] += face_normal

    # Normalize all vertex normals
    norms = np.linalg.norm(normals, axis=0)
    norms[norms < 1e-8] = 1  # avoid division by zero
    normals /= norms

    return normals




# # Example usage of the calc_normals function (uncomment the following lines to test):

# # # ----------------------------------------------------------------------------
# # # DEMO 1: Pyramid with square base
# # # ----------------------------------------------------------------------------


# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


# # Define vertex positions (3 x Nv)
# pts = np.array([
#     [0, 1, 1, 0, 0.5],  # x
#     [0, 0, 1, 1, 0.5],  # y
#     [0, 0, 0, 0, 1]     # z
# ], dtype=np.float64)

# # Define triangles by vertex indices (3 x Nt) - zero-indexed
# t_pos_idx = np.array([
#     [0, 1, 2, 3, 0, 1, 2, 3],  
#     [1, 2, 3, 0, 1, 2, 3, 0],
#     [2, 3, 0, 1, 4, 4, 4, 4]
# ], dtype=int)

# # Calculate normals
# normals = calc_normals(pts, t_pos_idx)

# # Plot the mesh and normals
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_title('Pyramid Vertex Normals')

# # Plot vertices
# ax.scatter(pts[0], pts[1], pts[2], color='black', label='Vertices')

# # Plot triangle faces
# for tri in t_pos_idx.T:
#     tri_pts = pts[:, tri]
#     tri_pts = np.hstack((tri_pts, tri_pts[:, 0:1]))  # loop back
#     ax.plot(tri_pts[0], tri_pts[1], tri_pts[2], color='gray')

# # Plot normals
# scale = 0.2
# for i in range(pts.shape[1]):
#     x, y, z = pts[:, i]
#     nx, ny, nz = normals[:, i]
#     ax.quiver(x, y, z, nx, ny, nz, length=scale, color='red')

# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.set_box_aspect([1,1,1])
# plt.legend()
# plt.show()