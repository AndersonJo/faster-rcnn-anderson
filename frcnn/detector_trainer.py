import math
from typing import Tuple, List

import numpy as np

from frcnn.anchor import ground_truth_anchors, to_absolute_coord
from frcnn.config import Config
from frcnn.iou import cal_iou
from frcnn.nms import non_max_suppression


class DetectorTrainer(object):

    def __init__(self, config: Config, class_mapping: dict):
        self.anchor_scales = config.anchor_scales
        self.anchor_ratios = config.anchor_ratios
        self.anchor_stride = config.anchor_stride

        self.min_overlap = config.detector_min_overlap
        self.max_overlap = config.detector_max_overlap

        self.class_mapping = class_mapping

    def __call__(self, rpn_cls_output: np.ndarray, rpn_reg_output: np.ndarray, image_data: dict, overlap_threshold=0.9):
        """
        :param rpn_cls_output: (batch, fh, fw, 9)
        :param rpn_reg_output: (batch, fh, fw, 36) and each regression is (tx, ty, tw, th)
        :param image_data: a dictionary of image meta data
        :param overlap_threshold : overlap threshold used for Non Max Suppression
        :return:
        """
        # Transform the shape of RPN outputs to (None, 4) regression
        # Transform relative coordinates (tx, ty, tw, th) to absolute coordinates (x, y, w, h)
        anchors, probs = self._transform_rpn(rpn_reg_output, rpn_cls_output)

        # Non Maximum Suppression
        anchors, probs = non_max_suppression(anchors, probs, overlap_threshold=overlap_threshold, max_box=300)

        # rescaled_width, rescaled_height = image_data['rescaled_width'], image_data['rescaled_height']
        # image = cv2.imread(image_data['image_path'])
        # image = cv2.resize(image, (int(rescaled_width/16), int(rescaled_height/16)))
        gt_anchors, classes = ground_truth_anchors(image_data, subsampling_stride=self.anchor_stride)

        self.generate_train_data(anchors, gt_anchors, classes)

        # for i in range(len(anchors)):
        #     g = anchors[i] * 16
        #     cv2.rectangle(image, (g[0], g[1]), (g[0] + 5, g[1] + 5), (255, 255, 0))
        # ipdb.set_trace()
        # self._transform_subsampled_gta(image_data, image)
        # cv2.imwrite('temp/' + image_data['filename'], image)

    def generate_train_data(self, anchors: np.ndarray, gt_anchors: np.ndarray, gt_classes: List[str]):
        """
        Generate training data for Detector Network
        :param anchors: picked anchors passed by Non Maximum Suppression (min_x, min_y, max_x, max_y)
        :param gt_anchors: ground-truth anchors (min_x, min_y], max_x, max_y)
        :param gt_classes: list of ground-truth class names -> ['tvmonitor', 'person']
        :return:
        """
        # Calculate IoUs
        ious = np.zeros((len(gt_anchors), len(anchors)))
        for i in range(len(gt_anchors)):
            gt_anchor = gt_anchors[i]
            ious[i, :] = [cal_iou(a, gt_anchor) for a in anchors]

        # Filter minimum IoUs
        loc_g, loc_a = np.where(ious > self.min_overlap)
        ious = ious[loc_g, loc_a]
        gt_anchors = gt_anchors[loc_g]
        gt_classes = np.array(gt_classes)[loc_g]
        anchors = anchors[loc_a]

        # Region Of Interests (min_x, min_y, w, h)
        rois = np.copy(anchors)
        w = anchors[:, 2] - anchors[:, 0]
        h = anchors[:, 3] - anchors[:, 1]
        rois[:, 2] = w
        rois[:, 3] = h

        # Classification Target Data : what is this object? car? or bicycle?
        class_target = np.zeros((len(rois), len(self.class_mapping)))
        loc_bg = np.where(ious < self.max_overlap)[0]
        loc_obj = np.where(ious >= self.max_overlap)[0]

        class_indices = [self.class_mapping[cls_name] for cls_name in gt_classes[loc_obj]]
        class_target[loc_obj, class_indices] = 1
        class_target[loc_bg, self.class_mapping['bg']] = 1

        import ipdb
        ipdb.set_trace()

    def _transform_rpn(self, rpn_reg_output: np.ndarray, rpn_cls_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
                anchors[0, :, :, cur_anchor] = X  # the center coordinate of anchor's width (37, 50)
                anchors[1, :, :, cur_anchor] = Y  # the center coordinate of anchor's height (37, 50)
                anchors[2, :, :, cur_anchor] = anchor_width  # anchor width <scalar value>
                anchors[3, :, :, cur_anchor] = anchor_height  # anchor height <scalar value>
                anchors[:, :, :, cur_anchor] = to_absolute_coord(anchors[:, :, :, cur_anchor], regr)

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
