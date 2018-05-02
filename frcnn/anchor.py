from typing import List, Tuple
import numpy as np


def ground_truth_anchors(image_data: dict, subsampling_stride: List[int] = None) -> Tuple[np.ndarray, list]:
    """
    Create ground-truth anchors with rescaled images

    :param image_data: image meta dictionary
    :param subsampling_stride: divided by strides [x, y].
    :return: ground truth anchors (min_gx, min_gy, max_gx, max_gy)
    """
    bboxes = image_data['objects']
    width, height = image_data['width'], image_data['height']
    rescaled_width, rescaled_height = image_data['rescaled_width'], image_data['rescaled_height']

    gta = np.zeros((len(bboxes), 4))
    classes = list()
    for i, (cls_name, x1, y1, x2, y2) in enumerate(bboxes):
        classes.append(cls_name)
        w_ratio = rescaled_width / float(width)
        h_ratio = rescaled_height / float(height)

        if subsampling_stride is not None:
            w_ratio /= subsampling_stride[0]
            h_ratio /= subsampling_stride[1]

        gta[i, 0] = int(round(x1 * w_ratio))
        gta[i, 1] = int(round(y1 * h_ratio))
        gta[i, 2] = int(round(x2 * w_ratio))
        gta[i, 3] = int(round(y2 * h_ratio))

    return gta, classes


##############################################################################################
# Coordinates Tools
##############################################################################################

def to_absolute_coord(anchors, regrs):
    """
    To Absolute Coordinates Function
    The method converts relative coordinates (tx, ty, tw, th)
    to absolute coordinates (x, y, w, h) with provided anchors.

    Input Tensors
        - anchors: (x_center, y_center, width, height)
        - regres: predicted regression outputs (x, y, w, h)

    -------------------------------------------------------------------------
    Refer to "to_relative_coord" function

        tx = (gt_cx - a_cx) / a_width
        ty = (gt_cy - a_cy) / a_height
        tw = np.log(g_width / a_width)
        th = np.log(g_height / a_height)
    -------------------------------------------------------------------------

    :return: absolute coordinates of the anchor (x, y, w, h)
    """
    # Anchor
    ax = anchors[0, :, :]
    ay = anchors[1, :, :]
    aw = anchors[2, :, :]
    ah = anchors[3, :, :]

    # Regression output with relative coordinates
    tx = regrs[0, :, :]
    ty = regrs[1, :, :]
    tw = regrs[2, :, :]
    th = regrs[3, :, :]

    cx = tx * aw + ax  # center coordinate of width
    cy = ty * ah + ay  # center coordinate of height

    w = np.exp(tw.astype(np.float64)) * aw  # width
    h = np.exp(th.astype(np.float64)) * ah  # height
    min_x = cx - w / 2.  # top left x coordinate of the anchor
    min_y = cy - h / 2.  # top left y coordinate of the anchor

    min_x = np.round(min_x)
    min_y = np.round(min_y)
    w = np.round(w)
    h = np.round(h)

    return np.stack([min_x, min_y, w, h])


def to_relative_coord(gta_coord: np.ndarray, anchor_coord: np.ndarray) -> np.ndarray:
    """
    Create regression target data of region proposal network
    :param gta_coord: ground-truth box coordinates [x_min, y_min, x_max, y_max]
    :param anchor_coord: anchor box coordinates [x_min, y_min, x_max, y_max]
    :return: regression target (t_x, t_y, t_w, t_h)
    """
    # gt_cx: the center x (the center of width) of ground-truth box (in a rescaled image)
    # gt_cy: the center y (the center of height) of ground-truth box (in a rescaled image)
    gt_cx = (gta_coord[0] + gta_coord[2]) / 2.
    gt_cy = (gta_coord[1] + gta_coord[3]) / 2.

    # a_cx: the center x (the center of width) of the anchor box (in a rescaled image)
    # a_cy: the center y (the center of height) of the anchor box (in a rescaled image)
    a_cx = (anchor_coord[0] + anchor_coord[2]) / 2.
    a_cy = (anchor_coord[1] + anchor_coord[3]) / 2.

    # a_width: the width value of the anchor
    # a_height: the height value of the anchor
    a_width = anchor_coord[2] - anchor_coord[0]
    a_height = anchor_coord[3] - anchor_coord[1]
    g_width = gta_coord[2] - gta_coord[0]
    g_height = gta_coord[3] - gta_coord[1]

    tx = (gt_cx - a_cx) / a_width
    ty = (gt_cy - a_cy) / a_height
    tw = np.log(g_width / a_width)
    th = np.log(g_height / a_height)

    return np.array([tx, ty, tw, th])
