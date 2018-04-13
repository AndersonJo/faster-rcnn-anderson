import argparse

import math


class Config:
    ###################################
    # Dataset
    ###################################
    data_path = '/data/VOCdevkit'

    ###################################
    # BaseNetwork
    ###################################
    # Base Network Name
    net_name = 'vgg16'

    ###################################
    # Region Proposal Network & Anchor
    ###################################
    # Thread
    n_thread = 1

    # anchor box scales
    anchor_scales = [128, 256, 512]

    # anchor box ratios
    anchor_ratios = [[1, 1],
                     [1. / math.sqrt(2), 2. / math.sqrt(2)],
                     [2. / math.sqrt(2), 1. / math.sqrt(2)]]

    # rescale input image
    is_rescale = True

    @property
    def n_anchor(self) -> int:
        return len(self.anchor_ratios) * len(self.anchor_scales)


def singleton_config(parser: argparse.Namespace = None) -> Config:
    if hasattr(singleton_config, 'singleton'):
        return singleton_config.singleton

    config = Config()
    if parser is not None:
        # Data
        config.data_path = parser.data_path
        config.fen_name = parser.net
        config.n_thread = parser.thread
        config.is_rescale = parser.rescale

    singleton_config.singleton = config
    return singleton_config.singleton
