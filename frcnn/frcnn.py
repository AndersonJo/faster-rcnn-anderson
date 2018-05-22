import os
import re
from typing import Tuple

import cv2
import numpy as np
from keras import Model

from frcnn.anchor import to_absolute_coord_np, apply_regression_to_xywh
from frcnn.classifier import ClassifierNetwork
from frcnn.config import Config
from frcnn.fen import FeatureExtractionNetwork
from frcnn.logging import get_logger
from frcnn.rpn import RegionProposalNetwork

logger = get_logger(__name__)


class FRCNN(object):
    CHECKPOINT_REGEX = 'model_(?P<step>\d+)_(?P<loss>\d*\.\d*)\.hdf5'

    def __init__(self, config: Config, class_mapping: dict, input_shape=(None, None, 3), train: bool = False):
        self.train = train

        fen = FeatureExtractionNetwork(config, input_shape=input_shape)
        rpn = RegionProposalNetwork(fen, config, train=train)
        clf = ClassifierNetwork(rpn, config, class_mapping, train=train, n_feature=config.fen_depth)

        self.fen = fen
        self.rpn = rpn
        self.clf = clf

        self.regr_std = config.clf_regr_std

        # Initialize ModelAll
        self._model_path = config.model_path

        if train:
            self._model_all = self._init_all_model()

    def _init_all_model(self) -> Model:
        """
        Initialize AllModel used for saving or loading the whole graph.
        """
        image_input = self.fen.image_input
        roi_input = self.clf.roi_input
        rpn_cls = self.rpn.tensors['rpn_cls']
        rpn_reg = self.rpn.tensors['rpn_reg']
        clf_cls = self.clf.tensors['clf_cls']
        clf_reg = self.clf.tensors['clf_reg']

        model = Model([image_input, roi_input], [rpn_cls, rpn_reg, clf_cls, clf_reg])
        model.compile(optimizer='sgd', loss='mae')

        return model

    @property
    def fen_model(self) -> Model:
        return self.fen.model

    @property
    def rpn_model(self) -> Model:
        return self.rpn.model

    @property
    def clf_model(self) -> Model:
        return self.clf.model

    def generate_anchors(self, rpn_cls_output: np.ndarray, rpn_reg_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        The method do the followings
            1. Create Anchors on the basis of feature maps
            2. Convert reg_output (1, fh, fw, 36) to (4, fh, fw, 9) for Non Maximum Suppression
            3. Convert regression (tx, ty, tw, th) to (gx, gy, gw, gh)

        RPN Input Tensors
            - rpn_cls_output: ex. (1, 37, 50,  9) where 37 and 50 can change
            - rpn_reg_output: ex. (1, 37, 50, 36) where 37 and 50 can change

        :param rpn_cls_output: RPN classification output
        :param rpn_reg_output: RPN regression output
        :return: a tuple of anchors and classification probability
            - anchors: (min_x, min_y, max_x, max_y)
            - probs: classification probabilities (object or background)
        """

        anchor_scales = self.clf.anchor_scales
        anchor_ratios = self.clf.anchor_ratios
        n_scales = len(anchor_scales)
        n_ratios = len(anchor_ratios)
        _, fh, fw, n_anchor = rpn_cls_output.shape  # shape example (1, 37, 50, 9)

        fen_anchors = np.zeros((4, fh, fw, n_anchor), dtype='int8')

        for anchor_ratio_idx in range(n_ratios):
            for anchor_scale_idx in range(n_scales):
                # anchor_width: Anchor's width on feature maps
                #               For example, the sub-sampling ratio of VGG-16 is 16.
                #               That is, the size of original image decrease in the ratio 6 to 1
                # anchor_height: Anchor's height
                anchor_scale = anchor_scales[anchor_scale_idx]
                anchor_ratio = anchor_ratios[anchor_ratio_idx]
                anchor_width = (anchor_scale * anchor_ratio[0]) / self.clf.anchor_stride[0]
                anchor_height = (anchor_scale * anchor_ratio[1]) / self.clf.anchor_stride[1]
                anc_idx = anchor_scale_idx + n_ratios * anchor_ratio_idx

                regr = rpn_reg_output[0, :, :, anc_idx * 4: anc_idx * 4 + 4]  # ex. (37, 50, 4)
                regr = np.transpose(regr, (2, 0, 1))  # (4, 37, 50)

                X, Y = np.meshgrid(np.arange(fw), np.arange(fh))
                fen_anchors[0, :, :, anc_idx] = X  # the center coordinate
                fen_anchors[1, :, :, anc_idx] = Y  # the center coordinate of anchor's height (37, 50)
                fen_anchors[2, :, :, anc_idx] = anchor_width  # anchor width <scalar value>
                fen_anchors[3, :, :, anc_idx] = anchor_height  # anchor height <scalar value>
                fen_anchors[:, :, :, anc_idx] = to_absolute_coord_np(fen_anchors[:, :, :, anc_idx], regr)

                # it makes sure that anchors' width and height are at least 1
                # Convert (w, h) to (max_x, max_y) by adding (min_x, min_y) to (w, h)
                # anchors become (min_x, min_y, max_x, max_y) <--- Important!!!
                fen_anchors[2, :, :, anc_idx] = np.maximum(0, fen_anchors[2, :, :, anc_idx])
                fen_anchors[3, :, :, anc_idx] = np.maximum(0, fen_anchors[3, :, :, anc_idx])
                fen_anchors[2, :, :, anc_idx] += fen_anchors[0, :, :, anc_idx]
                fen_anchors[3, :, :, anc_idx] += fen_anchors[1, :, :, anc_idx]

                # Limit the anchors within the feature maps.
                fen_anchors[0, :, :, anc_idx] = np.maximum(0, fen_anchors[0, :, :, anc_idx])
                fen_anchors[1, :, :, anc_idx] = np.maximum(0, fen_anchors[1, :, :, anc_idx])
                fen_anchors[2, :, :, anc_idx] = np.minimum(fw, fen_anchors[2, :, :, anc_idx])
                fen_anchors[3, :, :, anc_idx] = np.minimum(fh, fen_anchors[3, :, :, anc_idx])

        # A.transpose((0, 3, 1, 2)) : (4, 38 height, 50 widht, 9) -> (4, 9, 38 height, 50 width)
        # np.reshape(A.transpose((0, 3, 1, 2)), (4, -1)) : -> (4, 17100)
        fen_anchors = np.reshape(fen_anchors.transpose((0, 3, 1, 2)), (4, -1)).transpose((1, 0))  # (17100, 4)
        probs = rpn_cls_output.transpose((0, 3, 1, 2)).reshape((-1))  # (17100,)

        # Filter weird anchors
        min_x = fen_anchors[:, 0]  # predicted min_x (top left x coordinate of the anchor)
        min_y = fen_anchors[:, 1]  # predicted min_y (top left y coordinate of the anchor)
        max_x = fen_anchors[:, 2]  # predicted max_x (bottom right x coordinate of the anchor)
        max_y = fen_anchors[:, 3]  # predicted max_y (bottom right y coordinate of the anchor)

        outside_idxs = np.where((min_x - max_x > 0) | (min_y - max_y > 0))
        fen_anchors = np.delete(fen_anchors, outside_idxs, 0)
        probs = np.delete(probs, outside_idxs, 0)

        return fen_anchors, probs

    def clf_predict(self, inputs: np.ndarray, anchors: np.ndarray, clf_threshold: float = 0.7, meta=None):
        """
        :param inputs: when training, it is batch image. when inference, it is feature maps
        :param anchors: (None, (min_x, min_y, max_x, max_y))
        :param clf_threshold: exclude predicted output which have lower probabilities than clf_threshold
        :return:
        """
        cls_pred = list()
        reg_pred = list()
        rois = list()
        for roi in self._iter_rois(anchors):  # (min_x, min_y, max_x, max_y) -> (min_x, min_y, w, h)
            cls_p, reg_p = self.clf_model.predict_on_batch([inputs, roi])
            cls_pred.append(cls_p)
            reg_pred.append(reg_p)
            rois.append(roi)

        cls_pred = np.concatenate(cls_pred, axis=1)
        reg_pred = np.concatenate(reg_pred, axis=1)
        rois = np.concatenate(rois, axis=1)

        # Exclude background classfication output and the ones which have low probabilities.
        _bg = self.clf.class_mapping['bg']
        mask = (np.max(cls_pred, axis=2) > clf_threshold) & (np.argmax(cls_pred, axis=2) != _bg)

        cls_pred = cls_pred[mask]  # (None, n_class) with background
        reg_pred = reg_pred[mask]  # (None, (n_class-1) * 4)
        rois = rois[mask]

        # Get regressions as
        cls_indices = np.argmax(cls_pred, axis=1).reshape(-1)
        regs = np.zeros((len(cls_indices), 4), dtype=np.float32)
        for i, cls_idx in enumerate(cls_indices):
            regs[i] = reg_pred[i, cls_idx: cls_idx + 4]  # regs <- (tx, ty, th, tw)

        regs[:, 0] /= self.regr_std[0]
        regs[:, 1] /= self.regr_std[1]
        regs[:, 2] /= self.regr_std[2]
        regs[:, 3] /= self.regr_std[3]

        cxcycwch = apply_regression_to_xywh(regs, rois)  # gta_regs <- (g_x, g_y, g_w, g_h)
        # Convert (g_min_x, g_min_y, g_w, g_h) -> (min_x, min_y, max_x, max_y)
        cxcycwch[:, 0] -= cxcycwch[:, 2] / 2.
        cxcycwch[:, 1] -= cxcycwch[:, 3] / 2.
        cxcycwch[:, 2] += cxcycwch[:, 0]
        cxcycwch[:, 3] += cxcycwch[:, 1]
        anchors = cxcycwch

        # Inverse Anchor Stride
        anchors[:, 0] *= self.clf.anchor_stride[0]
        anchors[:, 1] *= self.clf.anchor_stride[1]
        anchors[:, 2] *= self.clf.anchor_stride[0]
        anchors[:, 3] *= self.clf.anchor_stride[1]

        return cls_indices, anchors

    def _iter_rois(self, anchors):
        """
        :param anchors: anchors filtered by NMS. (None, (min_x, min_y, max_x, max_y))
        :return rois (1, None, (min_x, min_y, w, h))
        """
        N = anchors.shape[0]
        n_roi = self.clf.n_roi

        rois = np.copy(anchors)
        rois[:, 2] = anchors[:, 2] - anchors[:, 0]  # width
        rois[:, 3] = anchors[:, 3] - anchors[:, 1]  # height
        rois = np.expand_dims(rois, axis=0)

        for i in range(N):
            sliced_rois = rois[:, i:i + n_roi]
            _n = sliced_rois.shape[1]

            if _n != n_roi:
                n_residual = n_roi - _n

                new_rois = np.zeros((1, n_roi, 4))
                new_rois[:, :_n] = sliced_rois
                new_rois[:, _n:] = rois[:, :n_residual]
                sliced_rois = new_rois

            yield sliced_rois

    def save(self, filepath=None):
        if filepath is None:
            filepath = self._model_path
        self._model_all.save_weights(filepath)
        logger.info('saved ' + filepath)

    def load_latest_model(self) -> Tuple[float, str]:
        checkpoints = list()
        for filename in os.listdir('checkpoints'):
            match = re.match(self.CHECKPOINT_REGEX, filename)
            if match is None:
                continue

            step = int(match.group('step'))
            checkpoints.append((step, filename))

        lat_step = 0
        lat_checkpoint = None

        if len(checkpoints) >= 1:
            checkpoints = sorted(checkpoints, key=lambda c: c[0])
            lat_step, lat_checkpoint = checkpoints[-1]

            self.load(os.path.join('checkpoints', lat_checkpoint))
            logger.info('loaded latest checkpoint - ' + lat_checkpoint)
        else:
            logger.info('no checkpoint')

        return lat_step, lat_checkpoint

    def load_most_accurate_model(self) -> Tuple[float, str]:
        checkpoints = list()
        for filename in os.listdir('checkpoints'):
            match = re.match(self.CHECKPOINT_REGEX, filename)
            if match is None:
                continue

            loss = float(match.group('loss'))
            checkpoints.append((loss, filename))

        lat_loss = None
        lat_checkpoint = None

        if len(checkpoints) >= 1:
            checkpoints = sorted(checkpoints, key=lambda c: c[0])
            lat_loss, lat_checkpoint = checkpoints[0]

            self.load(os.path.join('checkpoints', lat_checkpoint))
            logger.info('loaded most accurate checkpoint - ' + lat_checkpoint)
        else:
            logger.info('no checkpoint')

        return lat_loss, lat_checkpoint

    def load(self, filepath=None):
        if filepath is None:
            filepath = self._model_path

        if self.train:
            self._model_all.load_weights(filepath)
        else:
            self.rpn_model.load_weights(filepath, by_name=True)
            self.clf_model.load_weights(filepath, by_name=True)

        logger.info('load: ' + filepath)
