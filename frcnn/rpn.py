import math
from typing import List, Tuple
import numpy as np
import keras.backend as K
from keras import Model
from keras.layers import Conv2D
from keras.optimizers import Adam

from frcnn.config import Config
from frcnn.fen import FeatureExtractionNetwork


class RegionProposalNetwork(object):

    def __init__(self, fen: FeatureExtractionNetwork, config: Config):
        """
        # Region Proposal Network
        :param n_anchor: the number of anchors (usually 9)
        :param rpn_depth: the depth of the intermediate layer in Region Proposal Network (usually 256 or 512)
        """
        self.fen = fen

        # Initialize Region Proposal Network
        self.anchor_scales = config.anchor_scales
        self.anchor_ratios = config.anchor_ratios
        self.n_anchor = len(config.anchor_scales) * len(config.anchor_ratios)  # number of anchors
        self.anchor_stride = config.anchor_stride

        self.rpn_layer = None
        self.rpn_cls = None
        self.rpn_reg = None
        self.rpn_model = None
        self._init_rpn(config.rpn_depth)

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
        rpn_losses = [rpn_loss_cls(self.n_anchor), rpn_loss_regr(self.n_anchor)]
        rpn_losses = [self.classification_loss(self.n_anchor), rpn_loss_regr(self.n_anchor)]
        self.rpn_model.compile(Adam(lr=1e-5, name='rpn_adam'), loss=rpn_losses)

    def classification_loss(self, n_anchor: int):
        """
        :param n_anchor: the number of anchors
        :return: classification loss function for region proposal network
        """

        def log_loss(y_true, y_pred):
            y_true = y_true[:, :, :, :n_anchor]

            cross_entorpy = K.binary_crossentropy(y_true, y_pred)
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


def rpn_loss_cls(num_anchors):
    epsilon = 1e-4

    def rpn_loss_cls_fixed_num(y_true, y_pred):
        y_true = y_true[:, :, :, :num_anchors]
        crossentropy = K.binary_crossentropy(y_pred[:, :, :, :], y_true)
        loss = 1 * K.sum(y_true * crossentropy) / K.sum(epsilon + y_true)

        return loss

    return rpn_loss_cls_fixed_num


def rpn_loss_regr(num_anchors):
    epsilon = 1e-4

    def rpn_loss_regr_fixed_num(y_true, y_pred):
        x = y_true[:, :, :, 4 * num_anchors:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), K.tf.float32)
        result = K.sum(
            y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(
            epsilon + y_true[:, :, :, :4 * num_anchors])

        return result

    return rpn_loss_regr_fixed_num
