import os
import re
from typing import Tuple

import cv2
import numpy as np
from keras import Model

from frcnn.anchor import to_absolute_coord_np, apply_regression_to_roi
from frcnn.classifier import ClassifierNetwork
from frcnn.config import Config
from frcnn.fen import FeatureExtractionNetwork
from frcnn.logging import get_logger
from frcnn.nms import non_max_suppression
from frcnn.rpn import RegionProposalNetwork

logger = get_logger(__name__)


class FRCNN(object):
    CHECKPOINT_REGEX = 'model_(?P<step>\d+)_\d*\.\d*\.hdf5'

    def __init__(self, config: Config, class_mapping: dict, input_shape=(None, None, 3), train: bool = False):
        fen = FeatureExtractionNetwork(config, input_shape=input_shape)
        rpn = RegionProposalNetwork(fen, config, train=train)
        clf = ClassifierNetwork(rpn, config, class_mapping)

        self.fen = fen
        self.rpn = rpn
        self.clf = clf

        self._clf_reg_std = config.clf_regr_std

        # Initialize ModelAll
        self._model_path = config.model_path
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

    def generate_anchors(self, rpn_cls_output: np.ndarray, rpn_reg_output: np.ndarray,
                         overlap_threshold=0.9) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Anchors on feature maps and then Do Non Maximum Suppression.

        :param rpn_cls_output: (batch, fh, fw, 9)
        :param rpn_reg_output: (batch, fh, fw, 36) and each regression is (tx, ty, tw, th)
        :param overlap_threshold : used for Non Max Suppression
        :return
            - anchors: anchors picked by NMS. (None, (min_x, min_y, max_x, max_y))
            - probs: classification probability vector (is it object or not?)
        """
        # Transform the shape of RPN outputs to (None, 4) regression
        # Transform relative coordinates (tx, ty, tw, th) to absolute coordinates (x, y, w, h)
        anchors, probs = self._generate_anchors(rpn_reg_output, rpn_cls_output)

        # Non Maximum Suppression
        anchors, probs = non_max_suppression(anchors, probs, overlap_threshold=overlap_threshold, max_box=300)
        return anchors, probs

    def _generate_anchors(self, rpn_reg_output: np.ndarray,
                          rpn_cls_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        _, fh, fw, n_anchor = rpn_cls_output.shape  # shape example (1, 37, 50, 9)

        anchors = np.zeros((4, fh, fw, n_anchor), dtype='int8')

        cur_anchor = 0
        for anchor_size in anchor_scales:
            for anchor_ratio in anchor_ratios:
                # anchor_width: Anchor's width on feature maps
                #           For example, the sub-sampling ratio of VGG-16 is 16.
                #           That is, the size of original image decrease in the ratio 6 to 1
                # anchor_height: Anchor's height
                anchor_width = (anchor_size * anchor_ratio[0]) / self.clf.anchor_stride[0]
                anchor_height = (anchor_size * anchor_ratio[1]) / self.clf.anchor_stride[1]

                regr = rpn_reg_output[0, :, :, cur_anchor * 4: cur_anchor * 4 + 4]  # ex. (37, 50, 4)
                regr = np.transpose(regr, (2, 0, 1))  # (4, 37, 50)

                X, Y = np.meshgrid(np.arange(fw), np.arange(fh))
                anchors[0, :, :, cur_anchor] = X  # the center coordinate of anchor's width (37, 50)
                anchors[1, :, :, cur_anchor] = Y  # the center coordinate of anchor's height (37, 50)
                anchors[2, :, :, cur_anchor] = anchor_width  # anchor width <scalar value>
                anchors[3, :, :, cur_anchor] = anchor_height  # anchor height <scalar value>
                anchors[:, :, :, cur_anchor] = to_absolute_coord_np(anchors[:, :, :, cur_anchor], regr)

                # it makes sure that anchors' width and height are at least 1
                # Convert (w, h) to (max_x, max_y) by adding (min_x, min_y) to (w, h)
                # anchors become (min_x, min_y, max_x, max_y) <--- Important!!!
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
        anchors = np.reshape(anchors.transpose((0, 3, 1, 2)), (4, -1)).transpose((1, 0))  # (17100, 4)
        probs = rpn_cls_output.transpose((0, 3, 1, 2)).reshape((-1))  # (17100,)

        # Filter weird anchors
        min_x = anchors[:, 0]  # predicted min_x (top left x coordinate of the anchor)
        min_y = anchors[:, 1]  # predicted min_y (top left y coordinate of the anchor)
        max_x = anchors[:, 2]  # predicted max_x (bottom right x coordinate of the anchor)
        max_y = anchors[:, 3]  # predicted max_y (bottom right y coordinate of the anchor)

        idxs = np.where((min_x - max_x >= 0) | (min_y - max_y >= 0))
        anchors = np.delete(anchors, idxs, 0)
        probs = np.delete(probs, idxs, 0)

        return anchors, probs

    def clf_predict(self, batch_image: np.ndarray, anchors: np.ndarray, clf_threshold: float = 0.7, img_meta=None):
        """
        :param batch_image: (1, h, w, 3) image
        :param anchors: (None, (min_x, min_y, max_x, max_y))
        :param clf_threshold: exclude predicted output which have lower probabilities than clf_threshold
        :return:
        """
        cls_pred = list()
        reg_pred = list()
        rois = list()
        for roi in self._iter_rois(anchors):
            cls_p, reg_p = self.clf_model.predict_on_batch([batch_image, roi])
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

        regs[:, 0] /= self._clf_reg_std[0]
        regs[:, 1] /= self._clf_reg_std[1]
        regs[:, 2] /= self._clf_reg_std[2]
        regs[:, 3] /= self._clf_reg_std[3]

        gta_regs = apply_regression_to_roi(regs, rois)  # gta_regs <- (g_x, g_y, g_w, g_h)
        # Convert (g_x, g_y, g_w, g_h) -> (min_x, min_y, max_x, max_y)
        gta_regs[:, 2] = gta_regs[:, 0] + gta_regs[:, 2]  # max_x
        gta_regs[:, 3] = gta_regs[:, 1] + gta_regs[:, 3]  # max_y

        # Inverse Anchor Stride
        gta_regs[:, 0] *= self.clf.anchor_stride[0]
        gta_regs[:, 1] *= self.clf.anchor_stride[1]
        gta_regs[:, 2] *= self.clf.anchor_stride[0]
        gta_regs[:, 3] *= self.clf.anchor_stride[1]

        return cls_indices, gta_regs

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

    def debug_nms_images(self, anchors: np.ndarray, img_meta: dict):
        image = cv2.imread(img_meta['image_path'])
        image = cv2.resize(image, (img_meta['rescaled_width'], img_meta['rescaled_height']))

        ratio_x = img_meta['rescaled_width'] / img_meta['width']
        ratio_y = img_meta['rescaled_height'] / img_meta['height']

        for anchor in anchors:
            min_x = anchor[0] * 16
            min_y = anchor[1] * 16
            max_x = anchor[2] * 16 + min_x
            max_y = anchor[3] * 16 + min_y
            cx = (min_x + max_x) // 2
            cy = (min_y + max_y) // 2
            cv2.rectangle(image, (cx, cy), (cx + 5, cy + 5), (0, 0, 255))

        for obj in img_meta['objects']:
            min_x, min_y, max_x, max_y = obj[1:]
            min_x = int(min_x * ratio_x)
            max_x = int(max_x * ratio_x)
            min_y = int(min_y * ratio_y)
            max_y = int(max_y * ratio_y)

            cx = (min_x + max_x) // 2
            cy = (min_y + max_y) // 2
            cv2.rectangle(image, (cx, cy), (cx + 5, cy + 5), (255, 255, 0))

        cv2.imwrite(os.path.join('temp', img_meta['filename']), image)

    def save(self, filepath=None):
        if filepath is None:
            filepath = self._model_path
        self._model_all.save_weights(filepath)
        logger.info('saved ' + filepath)

    def load_latest(self) -> Tuple[int, str]:
        checkpoints = list()
        for filename in os.listdir('checkpoints'):
            match = re.match(self.CHECKPOINT_REGEX, filename)
            if match is None:
                continue

            step = int(match.group('step'))
            checkpoints.append((step, filename))
        checkpoints = sorted(checkpoints, key=lambda c: c[0])
        lat_step, lat_checkpoint = checkpoints[-1]

        self.load(os.path.join('checkpoints', lat_checkpoint))
        logger.info('loaded latest checkpoint - ' + lat_checkpoint)
        return lat_step, lat_checkpoint

    def load(self, filepath=None):
        if filepath is None:
            filepath = self._model_path
        self._model_all.load_weights(filepath)
        logger.info('loaded ' + filepath)
