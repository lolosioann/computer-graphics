import numpy as np
import pytest
from lookat import lookat

def test_lookat_identity_frame():
    eye = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 1.0, 0.0])
    target = np.array([0.0, 0.0, 1.0])

    R, t = lookat(eye, up, target)

    expected_R = np.eye(3)
    expected_t = eye.reshape(1, 3)

    np.testing.assert_array_almost_equal(R, expected_R)
    np.testing.assert_array_equal(t, expected_t)

def test_lookat_off_axis():
    eye = np.array([1.0, 2.0, 3.0])
    up = np.array([0.0, 1.0, 0.0])
    target = np.array([1.0, 2.0, 4.0])  # Looking along +Z

    R, t = lookat(eye, up, target)

    # z-axis of camera frame
    zc = target - eye
    zc /= np.linalg.norm(zc)

    # y-axis corrected to be orthogonal to z
    yc = up - np.dot(up, zc) * zc
    yc /= np.linalg.norm(yc)

    xc = np.cross(yc, zc)

    expected_R = np.stack((xc, yc, zc), axis=1)
    expected_t = eye.reshape(1, 3)

    np.testing.assert_array_almost_equal(R, expected_R)
    np.testing.assert_array_almost_equal(t, expected_t)

def test_lookat_non_unit_up():
    eye = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 10.0, 0.0])  # Non-unit up vector
    target = np.array([0.0, 0.0, 1.0])

    R, t = lookat(eye, up, target)

    expected_R = np.eye(3)
    expected_t = eye.reshape(1, 3)

    np.testing.assert_array_almost_equal(R, expected_R)
    np.testing.assert_array_equal(t, expected_t)

def test_lookat_shapes():
    eye = np.array([1.0, 2.0, 3.0])
    up = np.array([0.0, 1.0, 0.0])
    target = np.array([1.0, 2.0, 4.0])

    R, t = lookat(eye, up, target)

    assert R.shape == (3, 3)
    assert t.shape == (1, 3)
