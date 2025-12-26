import numpy as np
import pytest
from render_object import render_object

def test_render_object_triangle():
    # Define a simple triangle in 3D space
    v_pos = np.array([
        [-0.5, -0.5, 1.0],
        [0.5, -0.5, 1.0],
        [0.0,  0.5, 1.0]
    ])

    # Vertex colors: red, green, blue
    v_clr = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])

    # Triangle face (one triangle)
    t_pos_idx = np.array([[0, 1, 2]])

    # Camera and render parameters
    plane_h = 2
    plane_w = 2
    res_h = 100
    res_w = 100
    focal = 1.0
    eye = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 1.0, 0.0])
    target = np.array([0.0, 0.0, 1.0])

    # Render the image
    img = render_object(v_pos, v_clr, t_pos_idx, plane_h, plane_w, res_h, res_w, focal, eye, up, target)

    # Assertions
    assert isinstance(img, np.ndarray)
    assert img.shape == (res_h, res_w, 3)

    # Check that the image is not all black (i.e., something was rendered)
    assert np.any(img > 0.0)
