import numpy as np

from region_matcher.preprocessing import image_to_tensor, letterbox_square, resize_square


def test_resize_square_shape() -> None:
    image = np.zeros((20, 40, 3), dtype=np.uint8)
    assert resize_square(image, 32).shape == (32, 32, 3)


def test_letterbox_square_shape() -> None:
    image = np.zeros((20, 40, 3), dtype=np.uint8)
    assert letterbox_square(image, 32).shape == (32, 32, 3)


def test_image_to_tensor_shape() -> None:
    image = np.zeros((16, 24, 3), dtype=np.uint8)
    tensor = image_to_tensor(image)
    assert tuple(tensor.shape) == (3, 16, 24)
