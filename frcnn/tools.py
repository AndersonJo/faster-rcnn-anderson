from typing import Tuple

import cv2
import numpy as np


###############################################################################################
#
# Preprocessing Tools
#
###############################################################################################

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


def rescale_image(img: np.ndarray, resized_width: int, resized_height: int) -> np.ndarray:
    """
    Rescale an Image
    :param img: image array
    :param resized_width: width of the image
    :param resized_height:  height of the image
    :return: resized image
    """
    return cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)


def cal_fen_output_size(base_name: str, width: int, height: int) -> Tuple[int, int, int]:
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
