import cv2
import numpy as np

from t_shading import t_shading


def load_texture():
    # Adjust path as needed
    texture = cv2.imread(
        "/home/johnlolos/Coding/computer_graphics/src/texImg.jpg"
    )
    # texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
    return texture


def test_single_triangle():
    img = np.ones((500, 500, 3), dtype=np.float32)
    texture = load_texture()

    test_img = t_shading(
        img,
        vertices=[[20, 20], [250, 80], [300, 400]],
        uv=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        textImg=texture,
    )
    test_img = (test_img * 255).clip(0, 255).astype(np.uint8)

    cv2.imshow("single_triangle", test_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    assert 1 == 1


def test_multiple_triangles():
    img = np.ones((500, 500, 3), dtype=np.float32)
    texture = load_texture()

    test_img = t_shading(
        img,
        vertices=[[20, 20], [250, 80], [300, 400]],
        uv=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        textImg=texture,
    )
    test_img = t_shading(
        test_img,
        vertices=[[100, 100], [400, 200], [300, 300]],
        uv=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        textImg=texture,
    )
    test_img = (test_img * 255).clip(0, 255).astype(np.uint8)

    cv2.imshow("multiple_triangles", test_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    assert 1 == 1


def test_out_of_bounds_vertex():
    img = np.ones((500, 500, 3), dtype=np.float32)
    texture = load_texture()

    test_img = t_shading(
        img,
        vertices=[[1000, 1000], [400, 200], [300, 300]],
        uv=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        textImg=texture,
    )
    test_img = (test_img * 255).clip(0, 255).astype(np.uint8)

    cv2.imshow("out_of_bounds_vertex", test_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    assert 1 == 1
