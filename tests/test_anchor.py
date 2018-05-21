import itertools

from frcnn.anchor import to_relative_coord_np, apply_regression_to_roi, to_absolute_coord, to_absolute_coord_np
from frcnn.config import singleton_config
from frcnn.rpn_trainer import RPNTrainer
from frcnn.voc import PascalVocData
from tests import DATASET_ROOT_PATH
import numpy as np


def test_relative_and_absolute_anchor():
    gtas = np.array([[0, 0, 1, 1],
                     [0, 0, 1, 1],
                     [0, 0, 2, 5],
                     [5, 0, 7, 7],
                     [0, 10, 8, 100],
                     [10, 11, 15, 20]], dtype=np.float64)

    ancs = np.array([[0, 0, 1, 1],
                     [2, 1, 3, 3],
                     [4, 2, 10, 8],
                     [3, 3, 10, 15],
                     [2, 8, 5, 120],
                     [1, 1, 2, 2]], dtype=np.float64)
    relatives = to_relative_coord_np(gtas, ancs)

    gtas_pred = list()
    for anc, regr in zip(ancs, relatives):
        xywh = to_absolute_coord(anc, regr)
        xywh = list(xywh)
        xywh[0] -= xywh[2] / 2.
        xywh[1] -= xywh[3] / 2.
        xywh[2] += xywh[0]
        xywh[3] += xywh[1]
        gtas_pred.append(xywh)

    gtas_pred = np.array(gtas_pred)
    assert (gtas_pred == gtas).all()
