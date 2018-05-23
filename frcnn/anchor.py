import math
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

        gta[i, 0] = x1 * w_ratio
        gta[i, 1] = y1 * h_ratio
        gta[i, 2] = x2 * w_ratio
        gta[i, 3] = y2 * h_ratio

    return gta, classes


##############################################################################################
# Single Coordinates Tools
##############################################################################################
def to_relative_coord(gta_coord: np.ndarray, anchor_coord: np.ndarray) -> np.ndarray:
    """
    Create a "single" regression target data of region proposal network
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


def to_absolute_coord(anchor, regr):
    """
    :param anchor: (min_x, min_y, max_x, max_y)
    :param regr: (tx, ty, tw, th)
    :return: (cx, cy, w, h)
    """

    w = anchor[2] - anchor[0]
    h = anchor[3] - anchor[1]
    cx = anchor[0] + (w / 2)
    cy = anchor[1] + (h / 2)

    tx, ty, tw, th = regr

    g_cx = tx * w + cx
    g_cy = ty * h + cy
    g_w = np.exp(tw) * w
    g_h = np.exp(th) * h

    return g_cx, g_cy, g_w, g_h


# def apply_single_reg_to_roi(reg: np.ndarray, roi: np.ndarray):
#     """
#     Apply predicted regression output to rois
#     :param regs: batch of (tx, ty, tw, th) .. predicted regressions
#     :param rois: batch of (x, y, w, h) .. basically this is anchors gone through NMS.
#     :return: batch of (g_x, g_y, g_w, g_h)
#     """
#     cx = roi[0] + roi[2] / 2  # x_a + w_a/2 = cx_a
#     cy = roi[1] + roi[3] / 2  # y_a + h_a/2 = cy_a
#
#     g_cx = reg[0] * roi[2] + cx  # tx * w_a + cx_a
#     g_cy = reg[1] * roi[3] + cy  # ty * h_a + cy_a
#     g_w = np.exp(reg[2]) * roi[2]  # exp(tw) * w_a
#     g_h = np.exp(reg[3]) * roi[3]  # exp(th) * h_a
#
#     g_cx = np.round(g_cx).astype('int')
#     g_cy = np.round(g_cy).astype('int')
#     g_w = np.round(g_w).astype('int')
#     g_h = np.round(g_h).astype('int')
#
#     return np.stack([g_cx, g_cy, g_w, g_h], axis=-1)


##############################################################################################
# Multiple Coordinates Tools
##############################################################################################

def to_absolute_coord_np(anchors, regrs):
    """
    To Absolute Coordinates Function
    The method converts relative coordinates (tx, ty, tw, th)
    to absolute coordinates (x, y, w, h) with provided anchors.

    Input Tensors
        - anchors: (x_center, y_center, width, height)
        - regres: predicted regression outputs (tx, ty, tw, th)

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
    a_cx = anchors[0, :, :]
    a_cy = anchors[1, :, :]
    a_w = anchors[2, :, :]
    a_h = anchors[3, :, :]

    # Regression output with relative coordinates
    tx = regrs[0, :, :]
    ty = regrs[1, :, :]
    tw = regrs[2, :, :]
    th = regrs[3, :, :]

    g_cx = tx * a_w + a_cx  # center coordinate of width
    g_cy = ty * a_h + a_cy  # center coordinate of height
    g_w = np.exp(tw.astype(np.float64)) * a_w  # width
    g_h = np.exp(th.astype(np.float64)) * a_h  # height
    min_x = g_cx - g_w / 2.  # top left x coordinate of the anchor
    min_y = g_cy - g_h / 2.  # top left y coordinate of the anchor

    return np.stack([min_x, min_y, g_w, g_h])


def to_relative_coord_np(gta_coords: np.ndarray, anchor_coords: np.ndarray):
    """
    The method is different from `to_relative_coord` in the respect of performance.
    :param gta_coords: ground-truth box coordinates [[x_min, y_min, x_max, y_max], ...]
    :param anchor_coords: anchor box coordinates [[x_min, y_min, x_max, y_max], ...]
    :return: regression target [[t_x, t_y, t_w, t_h], ...]
    """
    g_w = gta_coords[:, 2] - gta_coords[:, 0]
    g_h = gta_coords[:, 3] - gta_coords[:, 1]
    a_w = anchor_coords[:, 2] - anchor_coords[:, 0]
    a_h = anchor_coords[:, 3] - anchor_coords[:, 1]

    g_cx = gta_coords[:, 0] + (g_w / 2)
    g_cy = gta_coords[:, 1] + (g_h / 2)
    cx = anchor_coords[:, 0] + (a_w / 2)  # center coordinate of width of the anchor box
    cy = anchor_coords[:, 1] + (a_h / 2)  # center coordinate of height of the anchor box

    tx = (g_cx - cx) / a_w
    ty = (g_cy - cy) / a_h
    tw = np.log(g_w / a_w)
    th = np.log(g_h / a_h)
    return np.stack([tx, ty, tw, th], axis=-1)


def apply_regression_to_rois(regs: np.ndarray, rois: np.ndarray):
    """
    Apply predicted regression output to rois
    :param regs: batch of (tx, ty, tw, th) .. predicted regressions
    :param rois: batch of (min_x, min_y, w, h) .. basically this is anchors gone through NMS.
    :return: batch of (g_cx, g_cy, g_w, g_h)
    """
    cx_a = rois[:, 0] + rois[:, 2] / 2  # x_a + w_a/2 = cx_a
    cy_a = rois[:, 1] + rois[:, 3] / 2  # y_a + h_a/2 = cy_a

    g_cx = regs[:, 0] * rois[:, 2] + cx_a  # tx * w_a + cx_a
    g_cy = regs[:, 1] * rois[:, 3] + cy_a  # ty * h_a + cy_a
    g_w = np.exp(regs[:, 2]) * rois[:, 2]  # exp(tw) * w_a
    g_h = np.exp(regs[:, 3]) * rois[:, 3]  # exp(th) * h_a

    # g_cx = np.round(g_cx).astype('int')
    # g_cy = np.round(g_cy).astype('int')
    # g_w = np.round(g_w).astype('int')
    # g_h = np.round(g_h).astype('int')

    return np.stack([g_cx, g_cy, g_w, g_h], axis=-1)

#
# tx = (gt_cx - a_cx) / a_width
#       ty = (gt_cy - a_cy) / a_height
#       tw = np.log(g_width / a_width)
#       th = np.log(g_height / a_height)
# def to_absolute_coord(anchor, regr):
#     """
#     :param anchor: (min_x, min_y, max_x, max_y)
#     :param regr: (tx, ty, tw, th)
#     :return: (cx, cy, w, h)
#     """
#
#     w = anchor[2] - anchor[0]
#     h = anchor[3] - anchor[1]
#     cx = anchor[0] + (w / 2)
#     cy = anchor[1] + (h / 2)
#
#     tx, ty, tw, th = regr
#
#     g_cx = tx * w + cx
#     g_cy = ty * h + cy
#     g_w = np.exp(tw) * w
#     g_h = np.exp(th) * h
#
#     return g_cx, g_cy, g_w, g_h
