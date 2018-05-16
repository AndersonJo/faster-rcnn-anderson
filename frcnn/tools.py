from typing import Tuple

import cv2
import numpy as np


def cal_rescaled_size(width: int, height: int, min_side: int = 600) -> Tuple[int, int, float]:
    """
    Calculates rescaled image size; which an side of the rectangle is longer than or equal to min_size
    :param width: image width
    :param height: image height
    :param min_side: minimum pixel size
    :return: (width, height)
    """

    def _rescale(small: int, big: int) -> Tuple[int, int, float]:
        rescaled_ratio = float(min_side) / small
        resized_big = rescaled_ratio * big
        resized_small = min_side
        return int(resized_small), int(resized_big), rescaled_ratio

    if width <= height:
        width, height, ratio = _rescale(width, height)
    else:
        height, width, ratio = _rescale(height, width)
    return width, height, ratio


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
        w, h = calculate_resnet50_output(width, height)
        return w, h, 512
    elif base_name == 'inception_v3':
        return int(width // 33.35), int(height // 33.35), 2048
    else:
        _msg = 'output calculataion method for {0} model is not implemented'.format(base_name)
        raise NotImplementedError(_msg)


def calculate_resnet50_output(width, height):
    def get_output_length(input_length):
        filter_sizes = [7, 3, 3, 3, 3]
        stride = 2
        padding = 2
        for filter_size in filter_sizes:
            input_length = (input_length - filter_size + 2 * padding) // stride

        return input_length

    return get_output_length(width), get_output_length(height)


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image[:, :, (2, 1, 0)]  # BGR -> RGB

    # Transpose the image -> (channel, height, widht)
    # image = np.transpose(image, (2, 0, 1))

    # Normalize the image
    image = image - 127
    return image.copy()


def denormalize_image(image: np.ndarray) -> np.ndarray:
    image = image[:, :, (2, 1, 0)]  # RGB -> BGR

    # Denormalize
    image += 127
    return image.copy()
