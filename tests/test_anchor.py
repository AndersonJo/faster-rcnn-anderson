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

    config = singleton_config()
    scales = [scale * np.array(config.anchor_ratios) for scale in config.anchor_scales]
    scales = np.array(scales).reshape(9, 2)

    _comb = [range(len(anchor_scales)), range(len(anchor_ratios))]

    for i, (anc_scale_idx, anc_rat_idx) in enumerate(itertools.product(*_comb)):
        anc_scale = config.anchor_scales[anc_scale_idx]
        anc_rat = config.anchor_ratios[anc_rat_idx]

        print(scales[i], 'xxxxxx', anc_scale * anc_rat[0], anc_scale * anc_rat[1])

        print()
