import math
from typing import List, Tuple
import numpy as np
import keras.backend as K
from keras import Model
from keras.layers import Conv2D
from keras.optimizers import Adam

from frcnn.fen import FeatureExtractionNetwork


class RegionProposalNetwork(object):

    def __init__(self, fen: FeatureExtractionNetwork, anchor_scales: List[int] = (128, 256, 512),
                 anchor_ratios: List[float] = ([1, 1], [1. / math.sqrt(2), 2. / math.sqrt(2)],
                                               [2. / math.sqrt(2), 1. / math.sqrt(2)]),
                 anchor_stride: List[int] = (16, 16), rpn_depth: int = 512):
        """
        # Region Proposal Network
        :param n_anchor: the number of anchors (usually 9)
        :param rpn_depth: the depth of the intermediate layer in Region Proposal Network (usually 256 or 512)
        """
        self.fen = fen

        # Initialize Region Proposal Network
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.n_anchor = len(anchor_scales) * len(anchor_ratios)  # number of anchors
        self.anchor_stride = anchor_stride

        self.rpn_layer = None
        self.rpn_cls = None
        self.rpn_reg = None
        self.rpn_model = None
        self._init_rpn(rpn_depth)

    def _init_rpn(self, rpn_depth: int):
        """
        This method initializes Region Proposal Network after base network (i.e. VGG16)
        :param rpn_depth:
        :return:
        """
        # intermediate layer assures that specific depth like 256, 512 is
        intermediate_layer = Conv2D(rpn_depth, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                    activation='relu', kernel_initializer='glorot_uniform',
                                    name='rpn_intermediat_layer')(self.fen.output)

        # It classify whether it is an object or just a background.
        # As it is simple binary classification, sigmoid function is used.
        classification = Conv2D(self.n_anchor, kernel_size=(1, 1), activation='sigmoid',
                                kernel_initializer='glorot_uniform',
                                name='rpn_classification')(intermediate_layer)

        # regression model predicts 4 spatial values for each of anchors; x_center, y_center, width, height
        regression = Conv2D(self.n_anchor * 4, kernel_size=(1, 1), activation='linear',
                            kernel_initializer='zero', name='rpn_regression')(intermediate_layer)

        self.rpn_layer = intermediate_layer
        self.rpn_cls = classification
        self.rpn_reg = regression
        self.rpn_model = Model(self.fen.input_img, outputs=[self.rpn_cls, self.rpn_reg])

        rpn_losses = [self.classification_loss(self.n_anchor), self.regression_loss(self.n_anchor)]
        self.rpn_model.compile(Adam(lr=1e-5), loss=rpn_losses)

    @staticmethod
    def classification_loss(n_anchor: int):
        """
        :param n_anchor: the number of anchors
        :return: classification loss function for region proposal network
        """

        def log_loss(y_true, y_pred):
            y_true = y_true[:, :, :, :n_anchor]

            cross_entorpy = y_true * K.binary_crossentropy(y_true, y_pred)
            loss = K.sum(cross_entorpy)
            return loss

        return log_loss

    @staticmethod
    def regression_loss(n_anchor: int, huber_delta: float = 1.):
        """
        :param n_anchor: the number of anchors
        :param huber_delta: ....
        :return: (x_center, y_center, width, height) * n_anchor
                 regression predictions for each of anchors
        """

        def smooth_l1(y_true, y_pred):
            y_true = y_true[:, :, :, 4 * n_anchor:]
            x = K.abs(y_true - y_pred)
            x = K.switch(x < huber_delta, 0.5 * x ** 2, x - 0.5 * huber_delta)
            loss = K.sum(x)
            return loss

        return smooth_l1

    @property
    def model(self) -> Model:
        return self.rpn_model


