import cv2
import numpy as np

from f_shading import f_shading


def test_single_triangle():
    # Mock image (white canvas)
    img = np.ones((500, 500, 3), dtype=np.float32)

    # Apply shading with vertex colors in [0, 1]
    test_img = f_shading(
        img,
        vertices=[[20, 20], [250, 80], [300, 400]],
        vcolors=[[0, 0, 1], [0, 1, 0], [1, 0, 0]],
    )
    # Scale the image to [0, 255] and convert to uint8
    test_img = (test_img * 255).clip(0, 255).astype(np.uint8)

    # Convert RGB to BGR for OpenCV display
    test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
    # Show image
    cv2.imshow("single_triangle", test_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    assert 1 == 1


def test_multiple_triangles():
    # Mock image (white canvas)
    img = np.ones((500, 500, 3), dtype=np.float32)

    # Apply shading with vertex colors in [0, 1]
    test_img = f_shading(
        img,
        vertices=[[20, 20], [250, 80], [300, 400]],
        vcolors=[[0, 0, 1], [0, 1, 0], [1, 0, 0]],
    )
    test_img = f_shading(
        test_img,
        vertices=[[100, 100], [400, 200], [300, 300]],
        vcolors=[[1, 0, 0], [1, 1, 1], [1, 0, 1]],
    )
    # Scale the image to [0, 255] and convert to uint8
    test_img = (test_img * 255).clip(0, 255).astype(np.uint8)

    # Convert RGB to BGR for OpenCV display
    test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
    # Show image
    cv2.imshow("multiple_triangles", test_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    assert 1 == 1


def test_out_of_bounds_vertex():
    # Mock image (white canvas)
    img = np.ones((500, 500, 3), dtype=np.float32)

    # Apply shading with vertex colors in [0, 1]
    test_img = f_shading(
        img,
        vertices=[[1000, 1000], [400, 200], [300, 300]],
        vcolors=[[1, 0, 0], [1, 1, 1], [1, 0, 1]],
    )
    # Scale the image to [0, 255] and convert to uint8
    test_img = (test_img * 255).clip(0, 255).astype(np.uint8)

    # Convert RGB to BGR for OpenCV display
    test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
    # Show image
    cv2.imshow("out_of_bounds_vertex", test_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    assert 1 == 1
