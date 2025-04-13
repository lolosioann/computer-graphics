import numpy as np
import pytest

from vector_interp import vector_interp


def test_linear_interp_y_dim():
    p1 = (255, 0)
    p2 = (255, 255)
    V1 = (1, 0)
    V2 = (1, 1)
    result = vector_interp(p1, p2, V1, V2, 0.4 * 255, dim=2)
    expected = np.array([1, 0.4])
    assert np.allclose(result, expected)


def test_linear_interp_x_dim():
    p1 = (0, 0)
    p2 = (255, 0)
    V1 = 0
    V2 = 1
    result = vector_interp(p1, p2, V1, V2, 0.5 * 255, dim=1)
    expected = np.array(0.5)
    assert np.allclose(result, expected)


def test_same_point_returns_V1():
    p1 = (1, 1)
    p2 = (1, 1)
    V1 = (5, 5)
    V2 = (10, 10)
    result = vector_interp(p1, p2, V1, V2, 1, dim=1)
    expected = np.array([5, 5])
    assert np.allclose(result, expected)


def test_dim_validation():
    with pytest.raises(ValueError):
        vector_interp((0, 0), (1, 1), 0, 1, 0.5, dim=3)


def test_edge_case_same_x_coord():
    p1 = (0, 0)
    p2 = (0, 1)
    V1 = 10
    V2 = 20
    result = vector_interp(p1, p2, V1, V2, 0, dim=1)
    expected = np.array(10)
    assert np.allclose(result, expected)


def test_edge_case_same_y_coord():
    p1 = (0, 0)
    p2 = (1, 0)
    V1 = 10
    V2 = 20
    result = vector_interp(p1, p2, V1, V2, 0, dim=2)
    expected = np.array(10)
    assert np.allclose(result, expected)


def test_point_outside_bounds():
    p1 = (0, 0)
    p2 = (1, 1)
    V1 = (0, 0)
    V2 = (255, 255)
    with pytest.raises(ValueError):
        vector_interp(p1, p2, V1, V2, 2, dim=1)
