import numpy as np
import pytest
from transforms import translate, rotate, compose, convert2affine

# ------------------Test cases for the translate function------------------

def test_translate_basic():
    t_vec = np.array([1, 2, 3])
    expected = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 2],
        [0, 0, 1, 3],
        [0, 0, 0, 1]
    ], dtype=np.float64)
    np.testing.assert_array_equal(translate(t_vec), expected)

def test_translate_zeros():
    t_vec = np.array([0, 0, 0])
    expected = np.eye(4, dtype=np.float64)
    np.testing.assert_array_equal(translate(t_vec), expected)

def test_translate_negative():
    t_vec = np.array([-5.5, 2.2, 0.0])
    expected = np.array([
        [1, 0, 0, -5.5],
        [0, 1, 0,  2.2],
        [0, 0, 1,  0.0],
        [0, 0, 0,  1.0]
    ])
    np.testing.assert_array_almost_equal(translate(t_vec), expected)

def test_translate_dtype_conversion():
    t_vec = [1, 2, 3]  # List input
    result = translate(t_vec)
    assert isinstance(result, np.ndarray)
    assert result.shape == (4, 4)


# ------------------Test cases for the rotate function------------------

def test_rotate_identity():
    axis = np.array([0, 0, 1])
    angle = 0
    center = np.array([0, 0, 0])
    result = rotate(axis, angle, center)
    np.testing.assert_array_almost_equal(result, np.eye(4))

def test_rotate_90deg_z_axis():
    axis = np.array([0, 0, 1])
    angle = np.pi / 2
    center = np.array([0, 0, 0])
    expected = np.array([
        [0, -1, 0, 0],
        [1,  0, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1]
    ])
    result = rotate(axis, angle, center)
    np.testing.assert_array_almost_equal(result, expected, decimal=6)

def test_rotate_about_point():
    axis = np.array([0, 0, 1])
    angle = np.pi
    center = np.array([1, 0, 0])
    result = rotate(axis, angle, center)
    point = np.array([2, 0, 0, 1])  # homogeneous
    rotated = result @ point
    expected = np.array([0, 0, 0, 1])  # should reflect across x=1
    np.testing.assert_array_almost_equal(rotated, expected, decimal=6)

def test_rotate_invalid_axis():
    with pytest.raises(ValueError):
        rotate(np.array([0, 0, 0]), np.pi, np.array([0, 0, 0]))

def test_rotate_invalid_shapes():
    with pytest.raises(ValueError):
        rotate(np.array([1, 2]), 0.5, np.array([0, 0, 0]))
    with pytest.raises(ValueError):
        rotate(np.array([1, 0, 0]), 0.5, np.array([1, 2]))


# ------------------Test cases for the compose function------------------

def test_compose_identity():
    I = np.eye(4)
    T = translate(np.array([2, 3, 4]))
    result = compose(T, I)
    np.testing.assert_array_equal(result, T)
    result2 = compose(I, T)
    np.testing.assert_array_equal(result2, T)

def test_compose_translation_then_rotation():
    T = translate(np.array([1, 0, 0]))
    R = rotate(np.array([0, 0, 1]), np.pi / 2)

    C = compose(T, R)

    # Apply to point (0,0,0)
    point = np.array([0, 0, 0, 1])
    transformed = np.matmul(C, point)
    expected = np.array([0, 1, 0, 1])  # Move to (1,0,0) then rotate 90° around z

    np.testing.assert_array_almost_equal(transformed, expected, decimal=6)

def test_compose_invalid_shapes():
    with pytest.raises(ValueError):
        compose(np.eye(3), np.eye(4))
    with pytest.raises(ValueError):
        compose(np.eye(4), np.zeros((5, 5)))

def test_composition_order_matters():
    T = translate(np.array([1, 0, 0]))
    R = rotate(np.array([0, 0, 1]), np.pi / 2)

    CR = compose(R, T)
    RC = compose(T, R)

    # They should not be equal
    with pytest.raises(AssertionError):
        np.testing.assert_array_almost_equal(CR, RC)


# ------------------Test cases for the convert2affine function------------------

def test_convert2affine_identity():
    R = np.eye(3)
    t = np.zeros(3)
    A = convert2affine(R, t)
    expected = np.eye(4)
    np.testing.assert_array_equal(A, expected)

def test_convert2affine_translation_only():
    R = np.eye(3)
    t = np.array([1.0, 2.0, 3.0])
    A = convert2affine(R, t)
    expected = np.eye(4)
    expected[:3, 3] = t
    np.testing.assert_array_equal(A, expected)

def test_convert2affine_rotation_only():
    theta = np.pi / 2
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
    t = np.zeros(3)
    A = convert2affine(R, t)
    expected = np.eye(4)
    expected[:3, :3] = R
    np.testing.assert_array_almost_equal(A, expected)

def test_convert2affine_rotation_translation():
    theta = np.pi / 2
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
    t = np.array([4.0, 5.0, 6.0])
    A = convert2affine(R, t)
    expected = np.eye(4)
    expected[:3, :3] = R
    expected[:3, 3] = t
    np.testing.assert_array_almost_equal(A, expected)

def test_convert2affine_invalid_rotation_shape():
    R = np.eye(2)  # Invalid shape
    t = np.zeros(3)
    with pytest.raises(ValueError, match="Rotation matrix must be 3x3."):
        convert2affine(R, t)

def test_convert2affine_invalid_translation_shape():
    R = np.eye(3)
    t = np.zeros(2)  # Invalid shape
    with pytest.raises(ValueError, match="Translation vector must be 3 elements."):
        convert2affine(R, t)
