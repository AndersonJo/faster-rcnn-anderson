from typing import Tuple, List

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


#######################################################################################################
# Intersection Over Union
#######################################################################################################
def cal_iou(box1: List[int, int, int, int], box2: List[int, int, int, int]) -> float:
    """
    Calculate Intersection Over Union between two bounding boxes.
     * (x1, y1) : the top left point of the bounding box
     * (x2, y2) : the bottom right point of the bounding box
    :param box1: a list of coordinates of box1 [x1, y1, x2, y2]
    :param box2: a list of coordinates of box2 [x1, y1, x2, y2]
    :return: IoU value
    """
    intxn_area = intersection(box1, box2)
    union_area = union(box1, box2, intxn_area)
    return float(intxn_area) / float(union_area + 1e-6)


def intersection(box1: List[int, int, int, int], box2: List[int, int, int, int]):
    x = max(box1[0], box2[0])
    y = max(box1[1], box2[1])
    w = min(box1[2], box2[2]) - x
    h = min(box1[3], box2[3]) - y
    if w < 0 or h < 0:
        return 0
    return w * h


def union(box1: List[int, int, int, int], box2: List[int, int, int, int], area_intersection: float = None):
    area_a = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_b = (box2[2] - box2[0]) * (box2[3] - box2[1])

    if area_intersection is None:
        area_intersection = intersection(box1, box2)

    area_union = area_a + area_b - area_intersection
    return area_union
