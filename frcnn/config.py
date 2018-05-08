import argparse

import math


class Config:
    ###################################
    # Dataset
    ###################################
    data_path = '/data/VOCdevkit'
    shuffle = True
    augment = True

    # rescale input image
    is_rescale = True

    ###################################
    # Anchors
    ###################################
    # anchor box scales
    anchor_scales = [128, 256, 512]

    # anchor box ratios
    anchor_ratios = [[1, 1],
                     [1. / math.sqrt(2), 2. / math.sqrt(2)],
                     [2. / math.sqrt(2), 1. / math.sqrt(2)]]

    # anchor stride of x and y
    anchor_stride = [16, 16]

    ###################################
    # BaseNetwork
    ###################################
    net_name = 'vgg16'

    ###################################
    # Region Proposal Network & Anchor
    ###################################
    # Depth
    rpn_depth = 512

    # overlaps for RPN
    rpn_min_overlap = 0.3
    rpn_max_overlap = 0.7

    ###################################
    # Region of Interests
    ###################################
    # number of ROIs at once
    n_roi = 16

    # ROI pooling size
    roi_pool_size = (7, 7)

    ###################################
    # Classifier Network
    ###################################
    clf_min_overlap = 0.1
    clf_max_overlap = 0.7

    ###################################
    # Save & Load
    ###################################
    model_path = 'model.hdf5'

    @property
    def n_anchor(self) -> int:
        return len(self.anchor_ratios) * len(self.anchor_scales)


def singleton_config(parser: argparse.Namespace = None) -> Config:
    if hasattr(singleton_config, 'singleton'):
        return singleton_config.singleton

    config = Config()
    if parser is not None:
        # Data
        config.data_path = parser.data
        config.fen_name = parser.net
        config.is_rescale = parser.rescale

    singleton_config.singleton = config
    return singleton_config.singleton
