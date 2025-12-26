from pathlib import Path

import cv2
import numpy as np

from render_img import render_img


def load_texture():
    # Compute path relative to this file
    texture_path = (
        Path(__file__).resolve().parent.parent / "src" / "texImg.jpg"
    )

    # Load the texture using OpenCV
    texture = cv2.imread(str(texture_path))
    if texture is None:
        raise FileNotFoundError(f"Texture not found at: {texture_path}")

    return texture


def test_render_img_flat():
    texImg = load_texture()
    faces = np.array([[0, 1, 2], [2, 3, 0], [2, 1, 4]])
    vertices = np.array(
        [[100, 200], [300, 400], [100, 100], [180, 470], [200, 360]]
    )
    vcolors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [0, 0, 1]])
    uvs = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [1, 1]])
    depth = np.array([2, 1, 0, 3, 0])
    shading = "f"  # flat shading
    img = render_img(faces, vertices, vcolors, uvs, depth, shading, texImg)
    img = (img * 255).clip(0, 255).astype(np.uint8)
    cv2.imshow("Flat Shading", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    assert 1 == 1


def test_render_img_textured():
    texImg = load_texture()
    faces = np.array([[0, 1, 2], [2, 3, 0], [2, 1, 4]])
    vertices = np.array(
        [[100, 200], [300, 400], [100, 100], [180, 470], [200, 360]]
    )
    vcolors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [0, 0, 1]])
    uvs = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.7, 0.5]])
    depth = np.array([2, 1, 0, 3, 0])
    shading = "t"  # textured shading
    img = render_img(
        faces, vertices, vcolors, uvs, depth, shading=shading, texImg=texImg
    )
    cv2.imshow("Textured Shading", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    assert img is not None
