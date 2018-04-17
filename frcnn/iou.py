from typing import List


def cal_iou(box1: List[int], box2: List[int]) -> float:
    """
    Calculates Intersection Over Union between two bounding boxes.
     * (x1, y1) : the top left point of the bounding box
     * (x2, y2) : the bottom right point of the bounding box
    :param box1: a list of coordinates [x_min, y_min, x_max, y_max]
    :param box2: a list of coordinates [x_min, y_min, x_max, y_max]
    :return: IoU value
    """
    intxn_area = intersection(box1, box2)
    union_area = union(box1, box2, intxn_area)
    return float(intxn_area) / float(union_area + 1e-6)


def intersection(box1: List[int], box2: List[int]):
    """
    Calculates intersection between the box coordinates.
    :param box1: a list of coordinates [x_min, y_min, x_max, y_max]
    :param box2: a list of coordinates [x_min, y_min, x_max, y_max]
    :return: intersection value
    """
    x = max(box1[0], box2[0])
    y = max(box1[1], box2[1])
    w = min(box1[2], box2[2]) - x
    h = min(box1[3], box2[3]) - y
    if w < 0 or h < 0:
        return 0
    return w * h


def union(box1: List[int], box2: List[int], area_intersection: float = None):
    """
    Calculates union between the box coordinates.
    :param box1: a list of coordinates [x_min, y_min, x_max, y_max]
    :param box2: a list of coordinates [x_min, y_min, x_max, y_max]
    :param area_intersection: if provided it reduces computation.
    :return: union value
    """
    area_a = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_b = (box2[2] - box2[0]) * (box2[3] - box2[1])

    if area_intersection is None:
        area_intersection = intersection(box1, box2)

    area_union = area_a + area_b - area_intersection
    return area_union
