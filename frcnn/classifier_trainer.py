import os

import cv2
import math
from typing import Tuple, List, Union

import numpy as np

from frcnn.anchor import ground_truth_anchors, to_relative_coord_np
from frcnn.config import Config
from frcnn.iou import cal_iou


class ClassifierTrainer(object):

    def __init__(self, config: Config, class_mapping: dict):
        self.anchor_scales = config.anchor_scales
        self.anchor_ratios = config.anchor_ratios
        self.anchor_stride = config.anchor_stride

        self.min_overlap = config.clf_min_overlap
        self.max_overlap = config.clf_max_overlap

        self.class_mapping = class_mapping
        self.n_class = len(class_mapping)
        self.n_roi = config.n_roi

    def next_batch(self, anchors: np.ndarray, image_data: dict, debug_image=False) -> Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[None, None, None]]:
        """
        :param anchors: (None, (x, y, w, h))
        :param image_data: a dictionary of image meta data
        :param overlap_threshold : overlap threshold used for Non Max Suppression
        :return:
            - (None, None, None) or (rois, cls_y, reg_y)
        """
        # Create ground-truth anchors
        gt_anchors, classes = ground_truth_anchors(image_data, subsampling_stride=self.anchor_stride)

        # Create target dataset for Classifier
        rois, cls_y, reg_y, loc_obj, loc_bg = self._generate_train_data(anchors, gt_anchors, classes)
        picked_indices, pos_indices, neg_indices = self._pick(cls_y)

        if picked_indices is None:
            return None, None, None

        if debug_image:
            self._debug_images(rois, pos_indices, neg_indices, image_data)

        rois = rois[:, picked_indices, :]
        cls_y = cls_y[:, picked_indices, :]
        reg_y = reg_y[:, picked_indices, :]

        return rois, cls_y, reg_y

    def _pick(self, cls_y: np.ndarray) -> Union[Tuple[None, None, None], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        :return: selected indices
        """

        bg_idx = self.class_mapping['bg']
        neg_indices = np.where(cls_y[0, :, bg_idx] == 1)[0]
        pos_indices = np.where(cls_y[0, :, bg_idx] == 0)[0]

        if not len(pos_indices) or not len(neg_indices):
            return None, None, None

        try:
            # replace=False means it does not allow duplicate choices
            neg_indices = np.random.choice(neg_indices, self.n_roi // 2, replace=False)
        except ValueError as e:
            # ValueError is raised when neg_indices is fewer than n_roi
            pass

        try:
            pos_indices = np.random.choice(pos_indices, self.n_roi - len(neg_indices), replace=False)
        except ValueError as e:
            # when pos_indices are not enough, duplicate choice is allowed
            _p_indices = np.random.choice(pos_indices, self.n_roi - len(neg_indices) - len(pos_indices), replace=True)
            pos_indices = np.concatenate([pos_indices, _p_indices], axis=0)

        picked_indices = np.concatenate([pos_indices, neg_indices], axis=0)
        return picked_indices, pos_indices, neg_indices

    def _generate_train_data(self, anchors: np.ndarray, gt_anchors: np.ndarray, gt_classes: List[str]) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate training data for Classifier Network
        :param anchors: picked anchors passed by Non Maximum Suppression (min_x, min_y, max_x, max_y)
        :param gt_anchors: ground-truth anchors (min_x, min_y, max_x, max_y)
        :param gt_classes: list of ground-truth class names -> ['tvmonitor', 'person']
        :return:
            - rois: Region of Interest over anchors. (1, None, (min_x, min_y, w, h))
            - cls_y: one-hot vectors over anchors (ex. [[1, 0, 0, ...], [0, 1, 0, ...], ...]). (1, None 21)
            - reg_y:
        """
        # Calculate IoUs
        n_gta = len(gt_anchors)
        ious = np.zeros((n_gta, len(anchors)))
        loc_best_ious = np.zeros(n_gta, dtype='int')

        for i in range(n_gta):
            gt_anchor = gt_anchors[i]
            ious[i, :] = [cal_iou(a, gt_anchor) for a in anchors]
            loc_best_ious[i] = np.argmax(ious[i, :])

        # Filter minimum IoUs
        loc_g, loc_a = np.where(ious > self.min_overlap)
        # best_ious = ious[np.arange(n_gta), loc_best_ious]
        ious = ious[loc_g, loc_a]
        n_ious = len(ious)

        gt_anchors = gt_anchors[loc_g]
        gt_classes = np.array(gt_classes)[loc_g]
        anchors = anchors[loc_a]

        ################################################################################
        # Region Of Interests
        #   - rois: [[min_x, min_y, w, h], ...]
        ################################################################################
        rois = np.copy(anchors)
        rois[:, 2] = anchors[:, 2] - anchors[:, 0]  # width
        rois[:, 3] = anchors[:, 3] - anchors[:, 1]  # height

        ################################################################################
        # Classfication Targets
        #   Classification one-hot target vectors : what is this object? car? or bicycle? as one-hot vectors
        #   - class_targets: [[0, 1, 0, ..., 0], [1, 0, 0, ..., 0], ...]
        ################################################################################
        loc_bg = np.where(ious < self.max_overlap)[0]
        loc_obj = np.where(ious >= self.max_overlap)[0]
        n_obj = len(loc_obj)

        cls_y = np.zeros((n_ious, self.n_class))
        cls_y[loc_bg, self.class_mapping['bg']] = 1
        class_obj_indices = None
        if n_obj:
            class_obj_indices = np.array([self.class_mapping[cls_name] for cls_name in gt_classes[loc_obj]])
            cls_y[loc_obj, class_obj_indices] = 1

        ################################################################################
        # Regression and Classification Targets
        #   - class_obj_indices: a list of target label indices that are over the maximum overlap
        #                        [1, 1, 1, 16, 16, 5, 5, 8, 8, 8]
        #
        #   if class_obj_indices = [1]
        #   - coords: [[0, 0, 0, 0, tx, ty, tw, th, 0, 0, ...], ...]
        #   - labels: [[0, 0, 0, 0,  1,  1,  1,  1, 0, 0, ...], ...]
        #   if class_obj_indices = [2]
        #   - coords: [[0, 0, 0, 0, 0, 0, 0, 0, tx, ty, tw, th, 0, 0, ...], ...]
        #   - labels: [[0, 0, 0, 0, 0, 0, 0, 0,  1,  1,  1,  1, 0, 0, ...], ...]
        ################################################################################
        coords = np.zeros((n_ious, 4 * self.n_class))
        labels = np.zeros((n_ious, 4 * self.n_class))

        if n_obj and class_obj_indices is not None:
            gt_obj_anchors = gt_anchors[loc_obj]
            obj_anchors = anchors[loc_obj]
            relative_t = to_relative_coord_np(gt_obj_anchors, obj_anchors)

            _loc_v = np.array([np.arange(idx * 4, idx * 4 + 4) for idx in class_obj_indices])
            coords[loc_obj, _loc_v.T] = relative_t.T
            labels[loc_obj, _loc_v.T] = 1

        coords[:, 0] *= 8.
        coords[:, 1] *= 8.
        coords[:, 2] *= 4.
        coords[:, 3] *= 4.

        # Classifer Model only uses coords part.
        reg_y = np.concatenate([labels, coords], axis=1)

        rois = np.expand_dims(rois, axis=0)
        cls_y = np.expand_dims(cls_y, axis=0)
        reg_y = np.expand_dims(reg_y, axis=0)

        return rois, cls_y, reg_y, loc_obj, loc_bg

    def _debug_images(self, rois, loc_obj, loc_bg, img_meta):
        image = cv2.imread(img_meta['image_path'])
        image = cv2.resize(image, (img_meta['rescaled_width'], img_meta['rescaled_height']))

        ratio_x = img_meta['rescaled_width'] / img_meta['width']
        ratio_y = img_meta['rescaled_height'] / img_meta['height']

        for roi in rois[0, loc_obj]:
            self._rectangle(image, roi, color=(255, 255, 0), thickness=3)

        for roi in rois[0, loc_bg]:
            self._rectangle(image, roi, color=(200, 200, 200), thickness=2)

        for obj in img_meta['objects']:
            min_x, min_y, max_x, max_y = obj[1:]
            min_x = int(min_x * ratio_x)
            max_x = int(max_x * ratio_x)
            min_y = int(min_y * ratio_y)
            max_y = int(max_y * ratio_y)

            cx = (min_x + max_x) // 2
            cy = (min_y + max_y) // 2
            cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 0, 255), thickness=1)

        cv2.imwrite(os.path.join('temp', img_meta['filename']), image)

    def _rectangle(self, image, roi, color: Tuple[int, int, int] = (0, 0, 255), thickness=1):
        min_x = roi[0] * 16
        min_y = roi[1] * 16
        max_x = roi[2] * 16 + min_x
        max_y = roi[3] * 16 + min_y

        cx = (min_x + max_x) // 2
        cy = (min_y + max_y) // 2
        cv2.rectangle(image, (cx - 5, cy - 5), (cx + 5, cy + 5), color, thickness=thickness)
