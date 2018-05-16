import itertools

from frcnn.config import singleton_config
from frcnn.rpn_trainer import RPNTrainer
from frcnn.voc import PascalVocData
from tests import DATASET_ROOT_PATH
import numpy as np


def test_anchor():
    config = singleton_config()
    anchor_ratios = config.anchor_ratios
    anchor_scales = config.anchor_scales
    n_ratio = len(anchor_ratios)

    _comb = [range(len(anchor_scales)), range(len(anchor_ratios))]

    for anc_scale_idx, anc_rat_idx in itertools.product(*_comb):
        z_pos = 4 * (anc_rat_idx + n_ratio * anc_scale_idx)
        print(anc_scale_idx, anc_rat_idx, z_pos)
