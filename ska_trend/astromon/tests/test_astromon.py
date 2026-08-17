import numpy as np

from ska_trend.astromon import data


def test_crop_box_interior():
    image = np.arange(100).reshape(10, 10)
    # crop a 3x3 window (half_size=1) centered on pixel (4, 5) -> column 4, row 5
    cropped, x0, y0 = data.crop_box(image, 4, 5, 1)
    assert (x0, y0) == (3, 4)
    assert cropped.shape == (3, 3)
    # the center pixel of the crop is image[5, 4]
    assert cropped[1, 1] == image[5, 4]


def test_crop_box_edge_is_clipped():
    image = np.arange(100).reshape(10, 10)
    # near the corner the window is clipped to the image bounds (no negative indices)
    cropped, x0, y0 = data.crop_box(image, 0, 0, 3)
    assert (x0, y0) == (0, 0)
    assert cropped.shape == (4, 4)
    assert cropped[0, 0] == image[0, 0]
