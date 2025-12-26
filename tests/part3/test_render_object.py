import numpy as np
import pytest
import sys
from pathlib import Path

# Add src/part3 to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "part3"))

from render_object import render_object
from MatPhong import MatPhong


def test_render_object_simple_triangle():
    """Test rendering a simple triangle with Gouraud shading."""
    # Simple triangle vertices
    v_pos = np.array([
        [0.0, 1.0, 0.5],  # x
        [0.0, 0.0, 0.866],  # y
        [2.0, 2.0, 2.0]   # z
    ], dtype=np.float64)

    # UV coordinates
    v_uvs = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 1.0]
    ], dtype=np.float64)

    # Triangle indices (0-based)
    t_pos_idx = np.array([[0, 1, 2]], dtype=int).T

    # Simple solid red texture
    tex = np.ones((64, 64, 3), dtype=np.float32)
    tex[:, :, 0] = 1.0  # Red
    tex[:, :, 1] = 0.0  # Green
    tex[:, :, 2] = 0.0  # Blue

    # Camera parameters
    eye = np.array([0.5, 0.5, 5.0])
    target = np.array([0.5, 0.5, 0.0])
    up = np.array([0.0, 1.0, 0.0])

    # Material
    mat = MatPhong(ka=0.2, kd=0.6, ks=0.2, n=20)

    # Light
    l_pos = np.array([[0.5, 0.5, 10.0]])
    l_int = np.array([[1.0, 1.0, 1.0]])
    l_amb = np.array([0.1, 0.1, 0.1])

    # Render with Gouraud shading
    img = render_object(
        v_pos=v_pos,
        v_uvs=v_uvs,
        t_pos_idx=t_pos_idx,
        tex=tex,
        plane_h=2,
        plane_w=2,
        res_h=100,
        res_w=100,
        focal=1.0,
        eye=eye,
        up=up,
        target=target,
        mat=mat,
        l_pos=l_pos,
        l_int=l_int,
        l_amb=l_amb,
        shader='gouraud'
    )

    # Check output shape
    assert img.shape == (100, 100, 3)

    # Check value range
    assert np.all(img >= 0.0)
    assert np.all(img <= 1.0)


def test_render_object_phong_shader():
    """Test rendering with Phong shading."""
    # Simple triangle
    v_pos = np.array([
        [0.0, 1.0, 0.5],
        [0.0, 0.0, 0.866],
        [2.0, 2.0, 2.0]
    ], dtype=np.float64)

    v_uvs = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float64)
    t_pos_idx = np.array([[0, 1, 2]], dtype=int).T

    tex = np.ones((64, 64, 3), dtype=np.float32) * 0.8

    eye = np.array([0.5, 0.5, 5.0])
    target = np.array([0.5, 0.5, 0.0])
    up = np.array([0.0, 1.0, 0.0])

    mat = MatPhong(ka=0.2, kd=0.6, ks=0.2, n=20)

    l_pos = np.array([[0.5, 0.5, 10.0]])
    l_int = np.array([[1.0, 1.0, 1.0]])
    l_amb = np.array([0.1, 0.1, 0.1])

    # Render with Phong shading
    img = render_object(
        v_pos=v_pos,
        v_uvs=v_uvs,
        t_pos_idx=t_pos_idx,
        tex=tex,
        plane_h=2,
        plane_w=2,
        res_h=100,
        res_w=100,
        focal=1.0,
        eye=eye,
        up=up,
        target=target,
        mat=mat,
        l_pos=l_pos,
        l_int=l_int,
        l_amb=l_amb,
        shader='phong'
    )

    # Check output
    assert img.shape == (100, 100, 3)
    assert np.all(img >= 0.0)
    assert np.all(img <= 1.0)


def test_render_object_invalid_shader():
    """Test that invalid shader type raises error."""
    v_pos = np.array([[0.0, 1.0, 0.5], [0.0, 0.0, 0.866], [2.0, 2.0, 2.0]], dtype=np.float64)
    v_uvs = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float64)
    t_pos_idx = np.array([[0, 1, 2]], dtype=int).T
    tex = np.ones((64, 64, 3), dtype=np.float32)

    eye = np.array([0.5, 0.5, 5.0])
    target = np.array([0.5, 0.5, 0.0])
    up = np.array([0.0, 1.0, 0.0])

    mat = MatPhong(ka=0.2, kd=0.6, ks=0.2, n=20)
    l_pos = np.array([[0.5, 0.5, 10.0]])
    l_int = np.array([[1.0, 1.0, 1.0]])
    l_amb = np.array([0.1, 0.1, 0.1])

    with pytest.raises(ValueError, match="Unknown shader type"):
        render_object(
            v_pos=v_pos,
            v_uvs=v_uvs,
            t_pos_idx=t_pos_idx,
            tex=tex,
            plane_h=2,
            plane_w=2,
            res_h=100,
            res_w=100,
            focal=1.0,
            eye=eye,
            up=up,
            target=target,
            mat=mat,
            l_pos=l_pos,
            l_int=l_int,
            l_amb=l_amb,
            shader='invalid_shader'
        )


def test_render_object_multiple_lights():
    """Test rendering with multiple light sources."""
    v_pos = np.array([[0.0, 1.0, 0.5], [0.0, 0.0, 0.866], [2.0, 2.0, 2.0]], dtype=np.float64)
    v_uvs = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float64)
    t_pos_idx = np.array([[0, 1, 2]], dtype=int).T
    tex = np.ones((64, 64, 3), dtype=np.float32) * 0.8

    eye = np.array([0.5, 0.5, 5.0])
    target = np.array([0.5, 0.5, 0.0])
    up = np.array([0.0, 1.0, 0.0])

    mat = MatPhong(ka=0.2, kd=0.6, ks=0.2, n=20)

    # Multiple lights
    l_pos = np.array([
        [0.5, 0.5, 10.0],
        [5.0, 5.0, 10.0],
        [-5.0, 5.0, 10.0]
    ])
    l_int = np.array([
        [0.5, 0.5, 0.5],
        [0.3, 0.3, 0.3],
        [0.3, 0.3, 0.3]
    ])
    l_amb = np.array([0.1, 0.1, 0.1])

    img = render_object(
        v_pos=v_pos,
        v_uvs=v_uvs,
        t_pos_idx=t_pos_idx,
        tex=tex,
        plane_h=2,
        plane_w=2,
        res_h=100,
        res_w=100,
        focal=1.0,
        eye=eye,
        up=up,
        target=target,
        mat=mat,
        l_pos=l_pos,
        l_int=l_int,
        l_amb=l_amb,
        shader='gouraud'
    )

    assert img.shape == (100, 100, 3)
    assert np.all(img >= 0.0)
    assert np.all(img <= 1.0)
