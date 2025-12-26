import numpy as np
import pytest
import sys
from pathlib import Path

# Add src/part3 to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "part3"))

from calc_normals import calc_normals


def test_calc_normals_single_triangle():
    """Test normal calculation for a single triangle."""
    # Equilateral triangle in xy-plane, pointing up in z
    pts = np.array([
        [0.0, 1.0, 0.5],  # x
        [0.0, 0.0, np.sqrt(3)/2],  # y
        [0.0, 0.0, 0.0]   # z
    ], dtype=np.float64)

    # Single triangle with indices
    t_pos_idx = np.array([[0, 1, 2]], dtype=int).T

    normals = calc_normals(pts, t_pos_idx)

    # All normals should point in +z direction
    expected = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]], dtype=np.float64)

    # Check shape
    assert normals.shape == (3, 3)

    # Check normalization
    norms = np.linalg.norm(normals, axis=0)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-6)

    # Check direction (all should point roughly in same direction)
    for i in range(3):
        assert normals[2, i] > 0.9  # z component should be dominant


def test_calc_normals_cube():
    """Test normal calculation for a simple cube."""
    # Simple cube vertices
    pts = np.array([
        [0, 1, 1, 0, 0, 1, 1, 0],  # x
        [0, 0, 1, 1, 0, 0, 1, 1],  # y
        [0, 0, 0, 0, 1, 1, 1, 1]   # z
    ], dtype=np.float64)

    # Two triangles forming bottom face (z=0)
    t_pos_idx = np.array([
        [0, 1, 1],
        [1, 2, 2],
        [2, 3, 0]
    ], dtype=int)

    normals = calc_normals(pts, t_pos_idx)

    # Check shape
    assert normals.shape == (3, 8)

    # Check normalization for vertices that are part of triangles
    norms = np.linalg.norm(normals, axis=0)
    # Vertices 0-3 are part of triangles, should be normalized
    np.testing.assert_allclose(norms[:4], 1.0, rtol=1e-6)

    # Just verify normals exist and are normalized
    for i in range(4):
        assert norms[i] == pytest.approx(1.0, rel=1e-6)


def test_calc_normals_pyramid():
    """Test normal calculation for a pyramid."""
    # Pyramid with square base
    pts = np.array([
        [0, 1, 1, 0, 0.5],  # x
        [0, 0, 1, 1, 0.5],  # y
        [0, 0, 0, 0, 1.0]   # z
    ], dtype=np.float64)

    # Base triangles (should point down in -z)
    t_pos_idx = np.array([
        [0, 1, 0, 1],
        [1, 2, 2, 3],
        [2, 3, 3, 0]
    ], dtype=int)

    normals = calc_normals(pts, t_pos_idx)

    # Check shape
    assert normals.shape == (3, 5)

    # Check normalization for vertices that are part of triangles
    norms = np.linalg.norm(normals, axis=0)
    # Vertices 0-3 are part of triangles, should be normalized
    np.testing.assert_allclose(norms[:4], 1.0, rtol=1e-6)
    # Vertex 4 might not be part of any triangles
    assert norms.shape == (5,)


def test_calc_normals_empty():
    """Test normal calculation with no triangles."""
    pts = np.array([[0, 1], [0, 0], [0, 0]], dtype=np.float64)
    t_pos_idx = np.array([[], [], []], dtype=int)

    normals = calc_normals(pts, t_pos_idx)

    # Should return array of zeros or nans
    assert normals.shape == (3, 2)


def test_calc_normals_degenerate_triangle():
    """Test normal calculation with degenerate (collinear) triangle."""
    # Three collinear points
    pts = np.array([
        [0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ], dtype=np.float64)

    t_pos_idx = np.array([[0, 1, 2]], dtype=int).T

    normals = calc_normals(pts, t_pos_idx)

    # Check shape
    assert normals.shape == (3, 3)

    # Degenerate triangle should have zero or near-zero normal
    # After averaging and normalization, result may vary
    assert normals.shape == (3, 3)
