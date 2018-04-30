import math
from typing import Tuple

import numpy as np
import cv2

from frcnn.config import singleton_config
from frcnn.nms import non_max_suppression_fast


class DetectorTrainer(object):

    def __init__(self):
        config = singleton_config()
        self.anchor_scales = config.anchor_scales
        self.anchor_ratios = config.anchor_ratios
        self.anchor_stride = config.anchor_stride

    def __call__(self, cls_output, reg_output):
        # nms_anchors: picked anchors (None, 4)
        nms_anchors, nms_regrs = self.rpn_to_roi(cls_output, reg_output)
        import ipdb
        ipdb.set_trace()
        # self.create_detector_target_data(nms_anchors, nms_regrs)

    def rpn_to_roi(self, rpn_cls_output: np.ndarray, rpn_reg_output: np.ndarray,
                   overlap_threshold=0.9) -> Tuple[np.ndarray, np.ndarray]:
        """
        The method do the followings
            - Create Anchors on the basis of feature maps
            - Prepare anchors and RPN outputs to `non max suppression`

        RPN Input Data
            - rpn_cls_output: ex. (1, 37, 50,  9) where 37 and 50 can change
            - rpn_reg_output: ex. (1, 37, 50, 36) where 37 and 50 can change

        TODO: optimization is required. the method takes more than 10 seconds.

        :param rpn_cls_output: RPN classification output
        :param rpn_reg_output: RPN regression output
        :param overlap_threshold : overlap threshold used for Non Max Suppression
        :return: anchors and regressions, processed by Non Max Suppression
        """
        anchor_scales = self.anchor_scales
        anchor_ratios = self.anchor_ratios
        _, fh, fw, n_anchor = rpn_cls_output.shape  # shape example (1, 37, 50, 9)

        anchors = np.zeros((4, fh, fw, n_anchor), dtype='int8')

        cur_anchor = 0
        for anchor_size in anchor_scales:
            for anchor_ratio in anchor_ratios:
                # anchor_width: Anchor's width on feature maps
                #           For example, the sub-sampling ratio of VGG-16 is 16.
                #           That is, the size of original image decrease in the ratio 6 to 1
                # anchor_height: Anchor's height
                anchor_width = (anchor_size * anchor_ratio[0]) / self.anchor_stride[0]
                anchor_height = (anchor_size * anchor_ratio[1]) / self.anchor_stride[1]

                regr = rpn_reg_output[0, :, :, cur_anchor * 4: cur_anchor * 4 + 4]  # ex. (37, 50, 4)
                regr = np.transpose(regr, (2, 0, 1))  # (4, 37, 50)

                X, Y = np.meshgrid(np.arange(fw), np.arange(fh))

                anchors[0, :, :, cur_anchor] = X  # the center coordinate of anchor's width
                anchors[1, :, :, cur_anchor] = Y  # the center coordinate of anchor's height
                anchors[2, :, :, cur_anchor] = anchor_width  # anchor width
                anchors[3, :, :, cur_anchor] = anchor_height  # anchor height
                anchors[:, :, :, cur_anchor] = self.reverse_rpn_regression(anchors[:, :, :, cur_anchor], regr)

                # it makes sure that anchors' width and height are at least 1
                anchors[2, :, :, cur_anchor] = np.maximum(1, anchors[2, :, :, cur_anchor])
                anchors[3, :, :, cur_anchor] = np.maximum(1, anchors[3, :, :, cur_anchor])
                anchors[2, :, :, cur_anchor] += anchors[0, :, :, cur_anchor]
                anchors[3, :, :, cur_anchor] += anchors[1, :, :, cur_anchor]

                # Limit the anchors within the feature maps.
                anchors[0, :, :, cur_anchor] = np.maximum(0, anchors[0, :, :, cur_anchor])
                anchors[1, :, :, cur_anchor] = np.maximum(0, anchors[1, :, :, cur_anchor])
                anchors[2, :, :, cur_anchor] = np.minimum(fw - 1, anchors[2, :, :, cur_anchor])
                anchors[3, :, :, cur_anchor] = np.minimum(fh - 1, anchors[3, :, :, cur_anchor])

                cur_anchor += 1

        # A.transpose((0, 3, 1, 2)) : (4, 38 height, 50 widht, 9) -> (4, 9, 38 height, 50 width)
        # np.reshape(A.transpose((0, 3, 1, 2)), (4, -1)) : -> (4, 17100)
        all_boxes = np.reshape(anchors.transpose((0, 3, 1, 2)), (4, -1)).transpose((1, 0))  # (17100, 4)
        all_probs = rpn_cls_output.transpose((0, 3, 1, 2)).reshape((-1))  # (17100,)

        x1 = all_boxes[:, 0]
        y1 = all_boxes[:, 1]
        x2 = all_boxes[:, 2]
        y2 = all_boxes[:, 3]

        idxs = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))

        all_boxes = np.delete(all_boxes, idxs, 0)
        all_probs = np.delete(all_probs, idxs, 0)

        nms_anchors, nms_regrs = non_max_suppression_fast(all_boxes, all_probs, overlap_threshold=overlap_threshold,
                                                          max_box=300)
        return nms_anchors, nms_regrs

    def create_detector_target_data(self):
        print('create_detector_target_data')

    @staticmethod
    def reverse_rpn_regression(anchors, regrs):
        """
        Find the ground-truth bounding box regression.

        -------------------------------------------------------------------------
        create_rpn_regression_target 함수를 보면 T 즉 rpn_reg_output은 다음과 같다.
        rpn_reg_output = [tx, ty, tw, th]

            tx = (gt_cx - a_cx) / a_width
            ty = (gt_cy - a_cy) / a_height
            tw = np.log(g_width / a_width)
            th = np.log(g_height / a_height)
        -------------------------------------------------------------------------

        :return : (g_x, g_y, g_w, g_h)
        """
        cx = anchors[0, :, :]
        cy = anchors[1, :, :]
        w = anchors[2, :, :]
        h = anchors[3, :, :]

        tx = regrs[0, :, :]
        ty = regrs[1, :, :]
        tw = regrs[2, :, :]
        th = regrs[3, :, :]

        gt_cx = tx * w + cx  # gt_cx -> center_w of ground-truth bounding box
        gt_cy = ty * h + cy  # gt_cy -> cneter_h of ground-truth bounding box

        g_w = np.exp(tw.astype(np.float64)) * w  # = g_width
        g_h = np.exp(th.astype(np.float64)) * h  # = g_height
        min_gx = gt_cx - g_w / 2.  # min_x of ground-truth bounding box
        min_gy = gt_cy - g_h / 2.  # min_y of ground-truth bounding box

        min_gx = np.round(min_gx)
        min_gy = np.round(min_gy)
        g_w = np.round(g_w)
        g_h = np.round(g_h)

        return np.stack([min_gx, min_gy, g_w, g_h])

    @staticmethod
    def apply_regr(x, y, w, h, tx, ty, tw, th):
        try:
            cx = x + w / 2.
            cy = y + h / 2.
            cx1 = tx * w + cx
            cy1 = ty * h + cy
            w1 = math.exp(tw) * w
            h1 = math.exp(th) * h
            x1 = cx1 - w1 / 2.
            y1 = cy1 - h1 / 2.
            x1 = int(round(x1))
            y1 = int(round(y1))
            w1 = int(round(w1))
            h1 = int(round(h1))

            return x1, y1, w1, h1

        except ValueError:
            return x, y, w, h
        except OverflowError:
            return x, y, w, h
        except Exception as e:
            print(e)
            return x, y, w, h
