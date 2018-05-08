from typing import Tuple

import numpy as np
from keras import Model

from frcnn.classifier import ClassifierNetwork
from frcnn.config import Config
from frcnn.fen import FeatureExtractionNetwork
from frcnn.rpn import RegionProposalNetwork


class FRCNN(object):
    def __init__(self, config: Config, class_mapping: dict, input_shape=(None, None, 3)):
        fen = FeatureExtractionNetwork(config, input_shape=input_shape)
        rpn = RegionProposalNetwork(fen, config)
        clf = ClassifierNetwork(rpn, config, class_mapping)

        self.fen = fen
        self.rpn = rpn
        self.clf = clf

        # Initialize ModelAll
        self._model_save_path = config.model_path
        self._model_all = self._init_all_model()

    def _init_all_model(self) -> Model:
        """
        Initialize AllModel used for saving or loading the whole graph.
        """
        image_input = self.fen.input_img
        roi_input = self.clf.roi_input
        rpn_cls = self.rpn.tensors['rpn_cls']
        rpn_reg = self.rpn.tensors['rpn_reg']
        clf_cls = self.clf.tensors['clf_cls']
        clf_reg = self.clf.tensors['clf_reg']

        model = Model([image_input, roi_input], [rpn_cls, rpn_reg, clf_cls, clf_reg])
        model.compile(optimizer='sgd', loss='mae')

        return model

    @property
    def fen_model(self) -> Model:
        return self.fen.model

    @property
    def rpn_model(self) -> Model:
        return self.rpn.model

    @property
    def clf_model(self) -> Model:
        return self.clf.model

    def save(self, filepath=None):
        if filepath is None:
            filepath = self._model_save_path
        self._model_all.save_weights(filepath)
