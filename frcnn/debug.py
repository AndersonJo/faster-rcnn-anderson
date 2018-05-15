from typing import List

import cv2

from frcnn.anchor import to_absolute_coord


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
        w, h = anc_scale * anc_rat[0], anc_scale * anc_rat[1]
        min_x = int(x_pos * 16 - w / 2)
        min_y = int(y_pos * 16 - h / 2)
        max_x = int(min_x + w)
        max_y = int(min_y + h)

        g_cx, g_cy, g_w, g_h = to_absolute_coord([min_x, min_y, max_x, max_y], reg_target)
        g_x1 = int(g_cx - g_w / 2)
        g_y1 = int(g_cy - g_h / 2)
        g_x2 = int(g_cx + g_w / 2)
        g_y2 = int(g_cy + g_h / 2)

        cv2.rectangle(_image, (g_x1, g_y1), (g_x2, g_y2), (0, 0, 255))
