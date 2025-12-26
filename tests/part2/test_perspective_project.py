import numpy as np
from perspective_project import perspective_project

def test_perspective_project_identity():
    pts = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 2.0]
    ])
    focal = 1.0
    R = np.eye(3)
    t = np.zeros(3)

    proj, depth = perspective_project(pts, focal, R, t)

    expected_proj = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0]
    ])
    expected_depth = np.array([1.0, 1.0, 1.0, 2.0])

    np.testing.assert_array_almost_equal(proj, expected_proj)
    np.testing.assert_array_almost_equal(depth, expected_depth)

def test_perspective_project_translation_only():
    pts = np.array([[0.0, 0.0, 2.0]])
    R = np.eye(3)
    t = np.array([0.0, 0.0, 1.0])  # Camera moved forward
    focal = 1.0

    proj, depth = perspective_project(pts, focal, R, t)

    expected_proj = np.array([[0.0, 0.0]])
    expected_depth = np.array([1.0])

    np.testing.assert_array_almost_equal(proj, expected_proj)
    np.testing.assert_array_almost_equal(depth, expected_depth)
