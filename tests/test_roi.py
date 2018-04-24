from frcnn.roi import non_max_suppression_fast
import numpy as np


def test_non_maximum_suppress():
    # box = (x_min, y_min, x_max, y_max)
    boxes = np.array([[3, 4, 10, 11]])

    non_max_suppression_fast(boxes, 0.5)
