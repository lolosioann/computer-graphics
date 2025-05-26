import numpy as np
import pytest
from rasterize import rasterize 

def test_rasterize_center():
    pts = np.array([[0.0, 0.0]])  
    plane_w, plane_h = 2, 2
    res_w, res_h = 4, 4

    pix = rasterize(pts, plane_w, plane_h, res_w, res_h)
    assert pix.shape == (1, 2)
    assert np.all(pix == np.array([[2, 2]]))


def test_rasterize_corners():
    pts = np.array([
        [-1.0, -1.0],  
        [1.0, 1.0],    
    ])
    plane_w, plane_h = 2, 2
    res_w, res_h = 4, 4

    pix = rasterize(pts, plane_w, plane_h, res_w, res_h)
    assert pix.shape == (2, 2)
    assert np.all(pix[0] == [0, 3])  
    assert np.all(pix[1] == [3, 0])  


def test_rasterize_clipping():
    pts = np.array([
        [-10.0, -10.0],
        [10.0, 10.0]
    ])
    plane_w, plane_h = 2, 2
    res_w, res_h = 5, 5

    pix = rasterize(pts, plane_w, plane_h, res_w, res_h)
    assert np.all(pix[0] == [0, 4])  
    assert np.all(pix[1] == [4, 0])  


def test_rasterize_multiple_points():
    pts = np.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [-0.5, -0.5],
        [1.0, -1.0]
    ])
    plane_w, plane_h = 2, 2
    res_w, res_h = 10, 10

    pix = rasterize(pts, plane_w, plane_h, res_w, res_h)
    assert pix.shape == (4, 2)
    assert np.all(pix[0] == [5, 5])
    assert np.all(pix[1] == [7, 2])
    assert np.all(pix[2] == [2, 7])
    assert np.all(pix[3] == [9, 9])
