from typing import Tuple

import cv2
import numpy as np


def cal_rescaled_size(width: int, height: int, min_side: int = 600) -> Tuple[int, int]:
    """
    Calculate rescaled image size; which an side of the rectangle is longer than or equal to min_size
    :param width: image width
    :param height: image height
    :param min_side: minimum pixel size
    :return: (width, height)
    """

    def _rescale(small: int, big: int) -> Tuple[int, int]:
        resized_big = float(min_side) / small * big
        resized_small = min_side
        return int(resized_small), int(resized_big)

    if width >= min_side and height >= min_side:
        pass

    elif width <= height:
        width, height = _rescale(width, height)
    else:
        height, width = _rescale(height, width)
    return width, height


def rescale_image(img: np.ndarray, resized_width: int, resized_height: int):
    """
    Rescale an Image
    :param img: image array
    :param resized_width: width of the image
    :param resized_height:  height of the image
    :return: resized image
    """
    return cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)


def get_anchor(dataset: list, rescale: bool = True, augment: bool = False):
    for voc in dataset:
        # Load Image
        image = cv2.imread(voc['image'])
        height, width, _ = image.shape

        # TODO: Write augmentation code
        if augment:
            pass

        # Rescale Image: at least one side of image should be larger than or equal to minimum size;
        # It may improve accuracy but decrease training or inference speed in trade-off.

        if rescale:
            resized_width, resized_height = cal_rescaled_size(width, height)
            image = rescale_image(image, resized_width, resized_height)
