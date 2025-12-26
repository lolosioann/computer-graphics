import numpy as np
import pytest
import sys
from pathlib import Path

# Add src/part3 to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "part3"))

from light import light
from MatPhong import MatPhong


def test_light_ambient_only():
    """Test lighting with only ambient component."""
    pt = np.array([0.0, 0.0, 0.0])
    nrm = np.array([0.0, 0.0, 1.0])
    vclr = np.array([0.8, 0.8, 0.8])
    cam_pos = np.array([0.0, 0.0, 5.0])

    # Zero diffuse and specular
    mat = MatPhong(ka=1.0, kd=0.0, ks=0.0, n=10)

    l_pos = np.array([[0.0, 0.0, 5.0]])
    l_int = np.array([[1.0, 1.0, 1.0]])
    l_amb = np.array([0.5, 0.5, 0.5])

    result = light(pt, nrm, vclr, cam_pos, mat, l_pos, l_int, l_amb)

    # Should only have ambient component: ka * vclr * l_amb
    expected = 1.0 * vclr * l_amb
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_light_diffuse_perpendicular():
    """Test diffuse lighting with light perpendicular to surface."""
    pt = np.array([0.0, 0.0, 0.0])
    nrm = np.array([0.0, 0.0, 1.0])
    vclr = np.array([1.0, 1.0, 1.0])
    cam_pos = np.array([0.0, 0.0, 5.0])

    # Only diffuse, no ambient or specular
    mat = MatPhong(ka=0.0, kd=1.0, ks=0.0, n=10)

    # Light directly above point
    l_pos = np.array([[0.0, 0.0, 5.0]])
    l_int = np.array([[1.0, 1.0, 1.0]])
    l_amb = np.array([0.0, 0.0, 0.0])

    result = light(pt, nrm, vclr, cam_pos, mat, l_pos, l_int, l_amb)

    # Diffuse should be maximum (cos(0) = 1)
    # kd * vclr * l_int * dot(nrm, L)
    assert result[0] > 0.9  # Should be close to 1.0


def test_light_specular_perfect_reflection():
    """Test specular lighting with perfect viewing angle."""
    pt = np.array([0.0, 0.0, 0.0])
    nrm = np.array([0.0, 0.0, 1.0])
    vclr = np.array([1.0, 1.0, 1.0])

    # Camera and light at same position above surface
    cam_pos = np.array([0.0, 0.0, 5.0])

    # Only specular
    mat = MatPhong(ka=0.0, kd=0.0, ks=1.0, n=10)

    l_pos = np.array([[0.0, 0.0, 5.0]])
    l_int = np.array([[1.0, 1.0, 1.0]])
    l_amb = np.array([0.0, 0.0, 0.0])

    result = light(pt, nrm, vclr, cam_pos, mat, l_pos, l_int, l_amb)

    # Specular should be strong (perfect reflection)
    assert result[0] > 0.5


def test_light_multiple_lights():
    """Test lighting with multiple light sources."""
    pt = np.array([0.0, 0.0, 0.0])
    nrm = np.array([0.0, 0.0, 1.0])
    vclr = np.array([1.0, 1.0, 1.0])
    cam_pos = np.array([0.0, 0.0, 5.0])

    mat = MatPhong(ka=0.1, kd=0.6, ks=0.3, n=20)

    # Three lights
    l_pos = np.array([
        [0.0, 0.0, 5.0],
        [5.0, 0.0, 5.0],
        [-5.0, 0.0, 5.0]
    ])
    l_int = np.array([
        [0.5, 0.5, 0.5],
        [0.3, 0.3, 0.3],
        [0.3, 0.3, 0.3]
    ])
    l_amb = np.array([0.1, 0.1, 0.1])

    result = light(pt, nrm, vclr, cam_pos, mat, l_pos, l_int, l_amb)

    # Result should be clipped to [0, 1]
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)

    # Should be brighter than with single light
    assert result[0] > 0.1


def test_light_output_range():
    """Test that light output is clamped to [0, 1]."""
    pt = np.array([0.0, 0.0, 0.0])
    nrm = np.array([0.0, 0.0, 1.0])
    vclr = np.array([1.0, 1.0, 1.0])
    cam_pos = np.array([0.0, 0.0, 5.0])

    # High coefficients to test clamping
    mat = MatPhong(ka=1.0, kd=1.0, ks=1.0, n=10)

    l_pos = np.array([[0.0, 0.0, 5.0]])
    l_int = np.array([[2.0, 2.0, 2.0]])  # Intensity > 1
    l_amb = np.array([1.0, 1.0, 1.0])

    result = light(pt, nrm, vclr, cam_pos, mat, l_pos, l_int, l_amb)

    # Should be clipped to [0, 1]
    np.testing.assert_array_less(result, 1.0 + 1e-6)
    np.testing.assert_array_less(-1e-6, result)


def test_light_colored_light():
    """Test lighting with colored light sources."""
    pt = np.array([0.0, 0.0, 0.0])
    nrm = np.array([0.0, 0.0, 1.0])
    vclr = np.array([1.0, 1.0, 1.0])
    cam_pos = np.array([0.0, 0.0, 5.0])

    mat = MatPhong(ka=0.2, kd=0.8, ks=0.0, n=10)

    # Red light
    l_pos = np.array([[0.0, 0.0, 5.0]])
    l_int = np.array([[1.0, 0.0, 0.0]])
    l_amb = np.array([0.0, 0.0, 0.0])

    result = light(pt, nrm, vclr, cam_pos, mat, l_pos, l_int, l_amb)

    # Red channel should be much stronger
    assert result[0] > result[1]
    assert result[0] > result[2]


def test_light_with_fixed_directions():
    """Test lighting with fixed V and L directions (Phong shading optimization)."""
    pt = np.array([0.0, 0.0, 0.0])
    nrm = np.array([0.0, 0.0, 1.0])
    vclr = np.array([1.0, 1.0, 1.0])
    cam_pos = np.array([0.0, 0.0, 5.0])

    mat = MatPhong(ka=0.1, kd=0.6, ks=0.3, n=20)

    l_pos = np.array([[0.0, 0.0, 5.0]])
    l_int = np.array([[1.0, 1.0, 1.0]])
    l_amb = np.array([0.1, 0.1, 0.1])

    # Compute fixed directions
    fixed_V = np.array([0.0, 0.0, 1.0])
    fixed_L = np.array([[0.0, 0.0, 1.0]])

    result = light(pt, nrm, vclr, cam_pos, mat, l_pos, l_int, l_amb,
                   fixed_V=fixed_V, fixed_L_list=fixed_L)

    # Result should be valid
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)
