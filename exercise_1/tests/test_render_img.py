import cv2
import numpy as np

from render_img import render_img

def test_render_flat_shading():
    # Define triangle
    vertices = np.array([[100, 100], [400, 120], [250, 400]])
    depth = np.array([0.2, 0.3, 0.1]).reshape(-1, 1)
    faces = np.array([[0, 1, 2]])
    v_colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # Call render_img with flat shading
    img = render_img(
        faces,
        vertices,
        v_colors,
        depth,
        shading="f",
    )

    # Display the result
    img_disp = (img * 255).clip(0, 255).astype(np.uint8)
    cv2.imshow("Flat Shading", img_disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_render_img_texture():
    # Define triangle
    vertices = np.array([[100, 100], [400, 120], [250, 400]])
    uv_coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
    depth = np.array([0.2, 0.3, 0.1]).reshape(-1, 1)
    faces = np.array([[0, 1, 2]])
    v_colors = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])

    # Load texture image
    text_img = cv2.imread(
        "/home/johnlolos/Coding/computer_graphics/src/texImg.jpg"
    )

    # Call render_img with texture shading
    img = render_img(
        faces,
        vertices,
        v_colors,
        depth,
        shading="t",
        text_img=text_img,
        uvs=uv_coords,
    )

    # Display the result
    img_disp = (img * 255).clip(0, 255).astype(np.uint8)
    cv2.imshow("Texture Shading", img_disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
