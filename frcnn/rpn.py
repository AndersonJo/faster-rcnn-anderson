from typing import List, Tuple
import numpy as np
import keras.backend as K


def create_rpn_regression_target(gta_coord: List[int], anchor_coord: List[int]) -> Tuple[float, float, float, float]:
    """
    Create regression target data of region proposal network
    :param gta_coord: ground-truth box coordinates [x_min, y_min, x_max, y_max]
    :param anchor_coord: anchor box coordinates [x_min, y_min, x_max, y_max]
    :return:
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

    # tx: distance/width 인데.. 비율이다. 만약에 distance그대로 사용하면 feature maps으로 줄어들었을때
    #     얼마만큼 차이가 나는지 모른다. width * tx 를 하면 나중에 얼마만큼 distance 차이가 나는지 알 수 있게 된다.
    # tx = (cx - cxa) / (x2_anc - x1_anc)
    # ty = (cy - cya) / (y2_anc - y1_anc)

    # log((ground-truth width)/(anchor width))
    # 나중에 np.exp(tw) * width 를 해주면 ground-truth width 값이 나온다.
    # apply_regr_np 함수를 찾아본다
    # tw = np.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc))
    # th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc))

    return tx, ty, tw, th


# Loss Functions

def rpn_classification_loss(n_anchor: int):
    """
    :param n_anchor: the number of anchors
    :return: classification loss function for region proposal network
    """

    def log_loss(y_true, y_pred):
        y_true = y_true[:, :, :, :n_anchor]

        cross_entorpy = y_true * K.binary_crossentropy(y_true, y_pred)
        loss = K.sum(cross_entorpy)
        return loss

    return log_loss


def rpn_regression_loss(n_anchor: int, huber_delta: float = 1.):
    """
    :param n_anchor: the number of anchors
    :param huber_delta: ....
    :return: (x_center, y_center, width, height) * n_anchor
             regression predictions for each of anchors
    """

    def smooth_l1(y_true, y_pred):
        y_true = y_true[:, :, :, 4 * n_anchor:]
        x = K.abs(y_true - y_pred)
        x = K.switch(x < huber_delta, 0.5 * x ** 2, x - 0.5 * huber_delta)
        loss = K.sum(x)
        return loss

    return smooth_l1
