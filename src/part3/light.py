from MatPhong import MatPhong
import numpy as np
from typing import Union, List, Optional

def light(
    pt: np.ndarray,                              # Surface point (3,)
    nrm: np.ndarray,                             # Surface normal at point (3,)
    vclr: np.ndarray,                            # Texture color at point (RGB in [0,1])
    cam_pos: np.ndarray,                         # Camera/viewer position (3,)
    mat: MatPhong,                               # Phong material
    l_pos: Union[np.ndarray, List[np.ndarray]],  # Light positions (N, 3) or list of (3,)
    l_int: Union[np.ndarray, List[np.ndarray]],  # Light intensities (N, 3)
    l_amb: np.ndarray,                           # Ambient light (RGB)
    fixed_V: Optional[np.ndarray] = None,        # Optional fixed view direction
    fixed_L_list: Optional[np.ndarray] = None    # Optional fixed light directions (N, 3)
) -> np.ndarray:

    """
    Computes Phong illumination at a point, optionally using fixed V and L directions.
    Use fixed_V and fixed_L_list in Phong shading to avoid per-pixel recalculation.
    """

    pt = np.asarray(pt).reshape(3)
    nrm = np.asarray(nrm).reshape(3)
    vclr = np.asarray(vclr).reshape(3)
    cam_pos = np.asarray(cam_pos).reshape(3)
    l_amb = np.asarray(l_amb).reshape(3)

    # Convert to list format if necessary
    if isinstance(l_pos, np.ndarray) and l_pos.ndim == 2:
        l_pos = [l_pos[i] for i in range(l_pos.shape[0])]
    if isinstance(l_int, np.ndarray) and l_int.ndim == 2:
        l_int = [l_int[i] for i in range(l_int.shape[0])]

    # Start with ambient component
    color = mat.ka * vclr * l_amb

    # Use fixed V if provided
    if fixed_V is not None:
        V = np.asarray(fixed_V).reshape(3)
    else:
        V = cam_pos - pt
        V = V / (np.linalg.norm(V) + 1e-8)

    for i, (lp, li) in enumerate(zip(l_pos, l_int)):
        lp = np.asarray(lp).reshape(3)
        li = np.asarray(li).reshape(3)

        # Use fixed L if provided
        if fixed_L_list is not None:
            L = np.asarray(fixed_L_list[i]).reshape(3)
        else:
            L = lp - pt
            L = L / (np.linalg.norm(L) + 1e-8)

        # Reflection vector
        R = 2 * np.dot(nrm, L) * nrm - L
        R = R / (np.linalg.norm(R) + 1e-8)

        # Diffuse
        diff = np.clip(np.dot(nrm, L), 0, 1)
        diffuse = mat.kd * vclr * li * diff

        # Specular
        spec = np.clip(np.dot(R, V), 0, 1) ** mat.n
        specular = mat.ks * li * spec

        # Add contributions
        color += diffuse + specular

    return np.clip(color, 0, 1)





# # Example usage of the light function with MatPhong material (uncomment the following lines to test):


# # ----------------------------------------------------------------------------
# # DEMO 1: Flat surface with one light source
# # ----------------------------------------------------------------------------

# # Dummy point and normal (a flat surface at origin, facing +z)
# pt = np.array([0.0, 0.0, 0.0])
# nrm = np.array([0.0, 0.0, 1.0])

# # Base color: light gray
# vclr = np.array([0.8, 0.8, 0.8])

# # Camera looking from z = 5
# cam_pos = np.array([0.0, 0.0, 5.0])

# # One light above at (0, 0, 5) with strong white intensity
# l_pos = np.array([[0.0, 0.0, 5.0]])
# l_int = np.array([[1.0, 1.0, 1.0]])

# # Ambient light (soft white)
# l_amb = np.array([0.1, 0.1, 0.1])

# # Phong material (moderate ambient, strong diffuse, some specular, shiny)
# mat = MatPhong(ka=0.3, kd=0.6, ks=0.5, n=10)

# # Call light function
# result_color = light(pt, nrm, vclr, cam_pos, mat, l_pos, l_int, l_amb)

# print("Resulting color:", result_color)




# # ----------------------------------------------------------------------------
# # DEMO 2: Multiple light sources
# # ----------------------------------------------------------------------------

# # Point on the surface and its normal vector
# pt = np.array([0.0, 0.0, 0.0])
# nrm = np.array([0.0, 0.0, 1.0])  # Normal pointing upward

# # Color at the surface point (base color)
# vclr = np.array([0.8, 0.8, 0.8])

# # Camera position (viewer located above the point)
# cam_pos = np.array([0.0, 0.0, 5.0])

# # 3 point light sources at different positions
# l_pos = np.array([
#     [0.0, 0.0, 5.0],    # Directly above the point
#     [5.0, 0.0, 5.0],    # To the right and above
#     [-5.0, 0.0, 5.0]    # To the left and above
# ])

# # Corresponding intensities (white light of varying strength)
# l_int = np.array([
#     [0.6, 0.6, 0.6],    # Strong
#     [0.5, 0.5, 0.5],    # Medium
#     [0.3, 0.3, 0.3]     # Weak
# ])

# # Ambient light intensity (global scene light)
# l_amb = np.array([0.1, 0.1, 0.1])

# # Material properties using the Phong model
# mat = MatPhong(ka=0.2, kd=0.6, ks=0.4, n=20)

# # Compute the resulting color at the point
# result_color = light(pt, nrm, vclr, cam_pos, mat, l_pos, l_int, l_amb)

# print("Resulting color with 3 light sources:", result_color)


# # ----------------------------------------------------------------------------
# # DEMO 3: Using fixed view and light directions (as used in Phong shading)
# # ----------------------------------------------------------------------------

# # Assume a centroid at the origin, and compute fixed vectors accordingly
# pt_center = np.array([1.0, 1.0, 0.0])

# # View direction (from centroid to camera)
# fixed_V = cam_pos - pt_center
# fixed_V = fixed_V / (np.linalg.norm(fixed_V) + 1e-8)

# # Light directions (from centroid to each light source)
# fixed_L_list = l_pos - pt_center
# fixed_L_list = fixed_L_list / (np.linalg.norm(fixed_L_list, axis=1, keepdims=True) + 1e-8)

# # Keep the point, normal, color same as in DEMO 2
# result_color_fixed = light(
#     pt=pt,                 # the surface point
#     nrm=nrm,               # normal at that point
#     vclr=vclr,             # texture color
#     cam_pos=cam_pos,
#     mat=mat,
#     l_pos=l_pos,
#     l_int=l_int,
#     l_amb=l_amb,
#     fixed_V=fixed_V,
#     fixed_L_list=fixed_L_list
# )

# print("Resulting color using fixed_V and fixed_L_list:", result_color_fixed)
