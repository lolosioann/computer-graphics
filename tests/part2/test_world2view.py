import numpy as np
import pytest
from world2view import world2view

def test_world2view_identity():
    pts = np.array([[1.0, 2.0, 3.0]])
    R = np.eye(3)
    c0 = np.zeros(3)

    out = world2view(pts, R, c0)
    np.testing.assert_array_almost_equal(out, pts)

def test_world2view_translation_only():
    pts = np.array([[1.0, 0.0, 0.0]])
    R = np.eye(3)
    c0 = np.array([1.0, 0.0, 0.0])

    out = world2view(pts, R, c0)
    expected = np.array([[0.0, 0.0, 0.0]])
    np.testing.assert_array_almost_equal(out, expected)

def test_world2view_rotation_only():
    pts = np.array([[1.0, 0.0, 0.0]])
    R = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0]
    ])
    c0 = np.zeros(3)

    out = world2view(pts, R, c0)
    expected = np.array([[0.0, -1.0, 0.0]])  # this is correct
    np.testing.assert_array_almost_equal(out, expected)

