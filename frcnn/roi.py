import math
from typing import List, Tuple

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import Input
from keras.engine import Layer
from keras.layers import TimeDistributed, Flatten, Dropout, Dense

from frcnn.model import RegionProposalNetwork
from frcnn.nms import non_max_suppression_fast


class RegionOfInterestPoolingLayer(Layer):

    def __init__(self, size: List[int] = (), n_roi: int = 32, method: str = 'resize', **kwargs):
        """
        See "Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition."

        :param size: the pool size of height and width.  
        :param n_roi: the number of regions of interest  
        """
        super(RegionOfInterestPoolingLayer, self).__init__(**kwargs)
        assert len(size) == 2

        self.pool_height = size[0]
        self.pool_width = size[1]
        self.n_roi = n_roi
        self.method = method
        self.n_channel = None

    def build(self, input_shape):
        super(RegionOfInterestPoolingLayer, self).build(input_shape)
        self.n_channel = input_shape[0][-1]

    def call(self, tensors: tf.Tensor, mask=None):
        """
        :param tensors
            - tensors[0] image: the convolution features of FEN (like VGG-16) -> (batch, height, width, features)
            - tensors[1] rois: RoI Input Tensor -> (batch, number of RoI, 4)
        :param mask: ...
        """
        image = tensors[0]  # ex. (?, ?, ?, 512)
        rois = tensors[1]  # ex. (?, 32, 4)

        outputs = list()
        for roi_idx in range(self.n_roi):
            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            # row_length = w / float(self.pool_width)
            # col_length = h / float(self.pool_height)

            x = K.cast(x, 'int32')
            y = K.cast(y, 'int32')
            w = K.cast(w, 'int32')
            h = K.cast(h, 'int32')

            # (None, 7 pool_height, 7 pool_width, 512 n_features)
            resized = tf.image.resize_images(image[:, y:y + h, x:x + w, :], (self.pool_height, self.pool_width))
            outputs.append(resized)

        final_output = K.concatenate(outputs, axis=0)
        final_output = K.reshape(final_output, (1, self.n_roi, self.pool_height, self.pool_width, self.n_channel))
        final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output

    def compute_output_shape(self, input_shape):
        return None, self.n_roi, self.pool_height, self.pool_width, self.n_channel


class ROINetwork(object):

    def __init__(self, rpn: RegionProposalNetwork, n_class: int = 20,
                 n_roi: int = 32, roi_pool_size: List[int] = (7, 7), roi_method: str = 'resize'):
        """
        # Region of Interest
        :param n_class: the number of classes (PASCAL VOC is 20)
        :param n_roi: the number of regions of interest
        :param roi_pool_size: size of roi pooling layer. (height, width)
        :param roi_method: ('resize', 'pooling') how to do region of interest
            - resize: this method is very simple but works properly
            - pooling: slow but it does what paper says to do exactly
        """
        self.rpn = rpn

        # Initiliaze Region of Interest
        assert len(roi_pool_size) == 2
        self.roi_input = Input(shape=(n_roi, 4))
        self.n_class = n_class  # number of classes (like car, human, bike, etc..)
        self.n_roi = n_roi
        self.roi_method = roi_method
        self.pool_height = roi_pool_size[0]
        self.pool_widht = roi_pool_size[1]
        self.roi_cls_output = None
        self.roi_reg_output = None
        self.roi_pooling_layer = RegionOfInterestPoolingLayer(size=roi_pool_size, n_roi=n_roi, method=roi_method)
        self._init_classifier()

    def _init_classifier(self) -> List[np.ndarray]:
        roi_pooled_output = self.roi_pooling_layer([self.rpn.fen.output, self.roi_input])

        h = TimeDistributed(Flatten(name='flatten'))(roi_pooled_output)
        h = TimeDistributed(Dense(4096, activation='relu', name='roi_fc1'))(h)
        h = TimeDistributed(Dropout(0.5))(h)
        h = TimeDistributed(Dense(4096, activation='relu', name='roi_fc2'))(h)
        h = TimeDistributed(Dropout(0.5))(h)

        cls_output = TimeDistributed(Dense(self.n_class, activation='softmax', kernel_initializer='zero'),
                                     name='roi_class_{}'.format(self.n_class))(h)

        reg_output = TimeDistributed(Dense(4 * (self.n_class - 1), activation='linear', kernel_initializer='zero'),
                                     name='roi_regress_{}'.format(self.n_class))(h)

        self.roi_cls_output = cls_output
        self.roi_reg_output = reg_output

        return [cls_output, reg_output]

    def to_roi(self, rpn_cls_output: np.ndarray, rpn_reg_output: np.ndarray,
               overlap_threshold=0.9) -> Tuple[np.ndarray, np.ndarray]:
        """
        The method do the followings
            - Create Anchors on the basis of feature maps
            - Prepare anchors and RPN outputs to `non max suppression`

        RPN Input Data
            - rpn_cls_output: ex. (1, 37, 50,  9) where 37 and 50 can change
            - rpn_reg_output: ex. (1, 37, 50, 36) where 37 and 50 can change

        :param rpn_cls_output: RPN classification output
        :param rpn_reg_output: RPN regression output
        :param overlap_threshold : overlap threshold used for Non Max Suppression
        :return: anchors and regressions, processed by Non Max Suppression
        """
        anchor_scales = self.rpn.anchor_scales
        anchor_ratios = self.rpn.anchor_ratios
        _, fh, fw, n_anchor = rpn_cls_output.shape  # shape example (1, 37, 50, 9)

        anchors = np.zeros((4, fh, fw, n_anchor), dtype='int8')

        cur_anchor = 0
        for anchor_size in anchor_scales:
            for anchor_ratio in anchor_ratios:
                # anchor_width: Anchor's width on feature maps
                #           For example, the sub-sampling ratio of VGG-16 is 16.
                #           That is, the size of original image decrease in the ratio 6 to 1
                # anchor_height: Anchor's height
                anchor_width = (anchor_size * anchor_ratio[0]) / self.rpn.anchor_stride[0]
                anchor_height = (anchor_size * anchor_ratio[1]) / self.rpn.anchor_stride[1]

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

        nms_anchors, nms_regrs = non_max_suppression_fast(all_boxes, all_probs, overlap_threshold=overlap_threshold)
        return nms_anchors, nms_regrs

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
