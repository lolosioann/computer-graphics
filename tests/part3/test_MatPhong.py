import sys
from pathlib import Path

import pytest

# Add src/part3 to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "part3"))

from MatPhong import MatPhong


def test_mat_phong_initialization():
    """Test MatPhong material initialization."""
    mat = MatPhong(ka=0.1, kd=0.6, ks=0.3, n=32)

    assert mat.ka == 0.1
    assert mat.kd == 0.6
    assert mat.ks == 0.3
    assert mat.n == 32


def test_mat_phong_zero_values():
    """Test MatPhong with zero coefficients."""
    mat = MatPhong(ka=0.0, kd=0.0, ks=0.0, n=1)

    assert mat.ka == 0.0
    assert mat.kd == 0.0
    assert mat.ks == 0.0
    assert mat.n == 1


def test_mat_phong_high_shininess():
    """Test MatPhong with high Phong exponent."""
    mat = MatPhong(ka=0.2, kd=0.5, ks=0.8, n=128)

    assert mat.n == 128


def test_mat_phong_all_ones():
    """Test MatPhong with all coefficients set to 1."""
    mat = MatPhong(ka=1.0, kd=1.0, ks=1.0, n=10)

    assert mat.ka == 1.0
    assert mat.kd == 1.0
    assert mat.ks == 1.0
    assert mat.n == 10


def test_mat_phong_attributes_exist():
    """Test that all expected attributes exist."""
    mat = MatPhong(ka=0.2, kd=0.6, ks=0.4, n=20)

    assert hasattr(mat, 'ka')
    assert hasattr(mat, 'kd')
    assert hasattr(mat, 'ks')
    assert hasattr(mat, 'n')
