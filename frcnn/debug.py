import os
from typing import List, Tuple

import cv2
import numpy as np

from frcnn.anchor import to_absolute_coord
from frcnn.config import singleton_config
from frcnn.tools import denormalize_image


def calculate_anchor_size():
    config = singleton_config()
    scales = list()
    for ratio_idx in range(len(config.anchor_ratios)):
        for scale_idx in range(len(config.anchor_scales)):
            z_pos = scale_idx + len(config.anchor_ratios) * ratio_idx
            anc_scale = config.anchor_scales[scale_idx]
            anc_ratio = np.array(config.anchor_ratios[ratio_idx])

            scales.append(anc_scale * anc_ratio)
    return np.array(scales)


def visualize_gta(image, meta, center=False):
    width_ratio = meta['rescaled_width'] / meta['width']
    height_ratio = meta['rescaled_height'] / meta['height']

    # Ground Truth
    for name, x1, y1, x2, y2 in meta['objects']:
        x1 *= width_ratio
        y1 *= height_ratio
        x2 *= width_ratio
        y2 *= height_ratio

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
        if center:
            cv2.rectangle(image, (cx - 3, cy - 3), (cx + 3, cy + 3), (0, 0, 255), thickness=2)


class RPNTrainerDebug:

    @classmethod
    def debug_next_batch(cls, image, meta, clsf, regr):
        config = singleton_config()

        # Calculate Scales
        scales = calculate_anchor_size()

        # Visualize GTA
        visualize_gta(image, meta)

        height, width, _ = image.shape
        image = denormalize_image(image)

        # Check Classification
        cls_h, cls_w, cls_o = np.where(np.logical_and(clsf[0, :, :, :9] == 1, clsf[0, :, :, 9:] == 1))
        regr = regr[0].copy()

        for i in range(len(cls_h)):
            loc_w = cls_w[i]
            loc_h = cls_h[i]
            loc_o = cls_o[i]

            cw = (loc_w + 0.5) * config.anchor_stride[0]
            ch = (loc_h + 0.5) * config.anchor_stride[1]

            anc_w, anc_h = scales[loc_o]

            cw = int(cw)
            ch = int(ch)
            cv2.rectangle(image, (cw, ch), (cw + 5, ch + 5), (0, 255, 255))

            min_x = cw - anc_w / 2
            min_y = ch - anc_h / 2
            max_x = cw + anc_w / 2
            max_y = ch + anc_h / 2

            min_x = int(min_x)
            min_y = int(min_y)
            max_x = int(max_x)
            max_y = int(max_y)

            tx, ty, tw, th = regr[loc_h, loc_w, (loc_o * 4) + 36:(loc_o * 4) + 4 + 36]
            g_cx, g_cy, g_w, g_h = to_absolute_coord([min_x, min_y, max_x, max_y], [tx, ty, tw, th])
            g_x1 = int(g_cx - g_w / 2)
            g_y1 = int(g_cy - g_h / 2)
            g_x2 = int(g_x1 + g_w)
            g_y2 = int(g_y1 + g_h)

            cv2.rectangle(image, (g_x1, g_y1), (g_x2, g_y2), (255, 255, 0), thickness=3)

        cv2.imwrite('temp/' + meta['filename'], image)


class TestRPN:
    """
    The class is used in `rpn_trainer.py -> RPNTargetProcessor -> generate_rpn_target`
    """

    @staticmethod
    def rectangle(image, x_pos: int, y_pos: int, anc_scale: int, anc_rat: List[float]):
        w, h = anc_scale * anc_rat[0], anc_scale * anc_rat[1]
        cv2.rectangle(image, (int(x_pos * 16 - w / 2), int(y_pos * 16 - h / 2)),
                      (int(x_pos * 16 + w / 2), int(y_pos * 16 + h / 2)),
                      (0, 0, 255))

    @staticmethod
    def point(image, x_pos: int, y_pos: int, color=(0, 0, 255)):
        cv2.rectangle(image, (x_pos * 16, y_pos * 16), (x_pos * 16 + 5, y_pos * 16 + 5), color)

    @staticmethod
    def apply(_image, x_pos, y_pos, anc_scale: int, anc_rat: List[float], reg_target):
        config = singleton_config()
        w, h = anc_scale * anc_rat[0], anc_scale * anc_rat[1]
        min_x = int(x_pos * config.anchor_stride[0] - w / 2)
        min_y = int(y_pos * config.anchor_stride[1] - h / 2)
        max_x = int(min_x + w)
        max_y = int(min_y + h)

        g_cx, g_cy, g_w, g_h = to_absolute_coord([min_x, min_y, max_x, max_y], reg_target)
        g_x1 = int(g_cx - g_w / 2)
        g_y1 = int(g_cy - g_h / 2)
        g_x2 = int(g_cx + g_w / 2)
        g_y2 = int(g_cy + g_h / 2)

        cv2.rectangle(_image, (g_x1, g_y1), (g_x2, g_y2), (0, 0, 255))


class FRCNNDebug:
    @staticmethod
    def debug_generate_anchors(image: np.ndarray, meta: dict, anchors: np.ndarray, probs,
                               cls_y: np.ndarray=None, reg_y: np.ndarray=None):
        """
        Anchors는 청생 포인트로 이미지에 점을 찍고, Ground-truth anchor는 빨간색 박스로 표시를 한다.
            - 빨간색 박스: meta에서 이미지에 대한 박스위치가 잘 잡혔는지 확인
            - 청색 포인트: anchors가 빨간색 박스 근처에서 잡혔는지 확인
        :param anchors:
        :param rescaled_image:
        :param meta:
        :return:
        """
        config = singleton_config()

        image = image.copy()
        image = denormalize_image(image).copy()
        ratio = meta['rescaled_ratio']

        # Visualize GTA
        visualize_gta(image, meta, center=True)

        for anchor, prob in zip(anchors, probs):
            min_x = (anchor[0] + 0.5) * config.anchor_stride[0]
            min_y = (anchor[1] + 0.5) * config.anchor_stride[1]
            max_x = (anchor[2] + 0.5) * config.anchor_stride[0]
            max_y = (anchor[3] + 0.5) * config.anchor_stride[1]
            cx = (min_x + max_x) / 2
            cy = (min_y + max_y) / 2

            min_x = int(min_x)
            min_y = int(min_y)
            max_x = int(max_x)
            max_y = int(max_y)
            cx = int(cx)
            cy = int(cy)

            if prob > 0.8:
                cv2.rectangle(image, (cx - 3, cy - 3), (cx + 3, cy + 3), (255, 255, 0), thickness=2)
                # cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (255 * prob, 255 * prob, 0), thickness=2)
            else:
                cv2.rectangle(image, (cx, cy), (cx + 5, cy + 5), (0, 0, 0), thickness=1)
                # cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 0, 0), thickness=1)

        cv2.imwrite(os.path.join('temp', meta['filename']), image)


class ClassifierDebug:

    @classmethod
    def debug_images(cls, rois, loc_obj, loc_bg, img_meta, image):

        image = denormalize_image(image.copy()).copy()

        ratio_x = img_meta['rescaled_width'] / img_meta['width']
        ratio_y = img_meta['rescaled_height'] / img_meta['height']

        for roi in rois[0, loc_obj]:
            cls._rectangle(image, roi, color=(255, 255, 0), thickness=2)

        for roi in rois[0, loc_bg]:
            cls._point(image, roi, color=(200, 200, 200), thickness=2)

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

    @classmethod
    def _rectangle(cls, image, roi, color: Tuple[int, int, int] = (0, 0, 255), thickness=1):
        min_x = roi[0] * 16
        min_y = roi[1] * 16
        max_x = roi[2] * 16 + min_x
        max_y = roi[3] * 16 + min_y

        cx = (min_x + max_x) // 2
        cy = (min_y + max_y) // 2
        cv2.rectangle(image, (min_x, min_y), (max_x, max_y), color, thickness=thickness)

    @classmethod
    def _point(cls, image, roi, color: Tuple[int, int, int] = (0, 0, 255), thickness=1):
        min_x = roi[0] * 16
        min_y = roi[1] * 16
        max_x = roi[2] * 16 + min_x
        max_y = roi[3] * 16 + min_y

        cx = (min_x + max_x) // 2
        cy = (min_y + max_y) // 2
        cv2.rectangle(image, (cx - 5, cy - 5), (cx + 5, cy + 5), color, thickness=thickness)


def check_clf_trainer_classification(cls_y, meta, inv_class_mapping):
    _answer_class = [obj[0] for obj in meta['objects']]
    if len(set(_answer_class)) >= 2:
        _pred_class = [inv_class_mapping[idx] for idx in np.argmax(cls_y, axis=2).tolist()[0]]
        _pred_class = list(filter(lambda x: x != 'bg', _pred_class))

        print()
        print('predict:', _pred_class)
        print('answer:', _answer_class)
        print()


def debug_nms_images(anchors: np.ndarray, meta: dict):
    image = cv2.imread(meta['image_path'])
    image = cv2.resize(image, (meta['rescaled_width'], meta['rescaled_height']))

    ratio_x = meta['rescaled_width'] / meta['width']
    ratio_y = meta['rescaled_height'] / meta['height']

    visualize_gta(image, meta)

    for anchor in anchors:
        min_x = anchor[0] * 16
        min_y = anchor[1] * 16
        max_x = anchor[2] * 16 + min_x
        max_y = anchor[3] * 16 + min_y
        cx = (min_x + max_x) // 2
        cy = (min_y + max_y) // 2
        cv2.rectangle(image, (cx, cy), (cx + 5, cy + 5), (0, 0, 255))

    cv2.imwrite(os.path.join('temp', meta['filename']), image)
