from typing import Tuple, List

import cv2
import numpy as np


def cal_rescaled_size(width: int, height: int, min_side: int = 600) -> Tuple[int, int]:
    """
    Calculates rescaled image size; which an side of the rectangle is longer than or equal to min_size
    :param width: image width
    :param height: image height
    :param min_side: minimum pixel size
    :return: (width, height)
    """

    def _rescale(small: int, big: int) -> Tuple[int, int]:
        resized_big = float(min_side) * big / small
        resized_small = min_side
        return int(resized_small), int(resized_big)

    if width <= height:
        width, height = _rescale(width, height)
    else:
        height, width = _rescale(height, width)
    return width, height


def rescale_image(img: np.ndarray, resized_width: int, resized_height: int) -> np.ndarray:
    """
    Rescales an Image
    :param img: image array
    :param resized_width: width of the image
    :param resized_height:  height of the image
    :return: resized image
    """
    return cv2.resize(img, (resized_width, resized_height))


def cal_fen_output_size(base_name: str, width: int, height: int) -> Tuple[int, int, int]:
    """
    Calculates the output size of Feature Extraction Network (A.K.A Base Network like VGG-16)
    :param base_name: network name ('vgg16', 'vgg19', 'resnet50', 'inception_v3')
    :param width: input width
    :param height: input height
    :return: output width and height of FEN
    """
    if base_name == 'vgg16':
        return width // 16, height // 16, 512
    elif base_name == 'vgg19':
        return width // 16, height // 16, 512
    elif base_name == 'resnet50':
        return int(width // 31.19), int(height // 31.19), 2048
    elif base_name == 'inception_v3':
        return int(width // 33.35), int(height // 33.35), 2048
    else:
        _msg = 'output calculataion method for {0} model is not implemented'.format(base_name)
        raise NotImplementedError(_msg)
