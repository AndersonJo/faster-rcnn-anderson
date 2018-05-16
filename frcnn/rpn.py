import keras.backend as K
import tensorflow as tf
from keras import Model
from keras.layers import Conv2D
from keras.optimizers import Adam

from frcnn.config import Config
from frcnn.fen import FeatureExtractionNetwork


class RegionProposalNetwork(object):

    def __init__(self, fen: FeatureExtractionNetwork, config: Config, train: bool = False):
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
        self.tensors = dict()
        self._init_rpn(config.rpn_depth, train)

    def _init_rpn(self, rpn_depth: int, train: bool = False):
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

        self.tensors['rpn_intm'] = intermediate_layer
        self.tensors['rpn_cls'] = classification
        self.tensors['rpn_reg'] = regression

        self.rpn_layer = intermediate_layer
        self.rpn_cls = classification
        self.rpn_reg = regression

        if train:
            self.rpn_model = Model(self.fen.image_input, outputs=[self.rpn_cls, self.rpn_reg])
            rpn_losses = [self.classification_loss(self.n_anchor), self.regression_loss(self.n_anchor)]
            self.rpn_model.compile(Adam(lr=1e-5), loss=rpn_losses)
        else:
            self.rpn_model = Model(self.fen.image_input, outputs=[self.rpn_cls, self.rpn_reg, self.fen.output])

    def classification_loss(self, n_anchor: int, lambda_cls: float = 0.8, epsilon: float = 1e-9):
        """
        :param n_anchor: the number of anchors
        :param epsilon: ...
        :return: classification loss function for region proposal network
        """

        def log_loss(y_true, y_pred):
            y_true = y_true[:, :, :, :n_anchor]

            cross_entorpy = K.binary_crossentropy(y_pred, y_true)
            loss = K.sum(y_true * cross_entorpy) / (K.sum(y_true) + epsilon)

            self.tensors['cls_y_true'] = y_true
            self.tensors['cls_y_pred'] = y_pred
            self.tensors['cls_cross_entorpy'] = cross_entorpy
            self.tensors['cls_loss'] = loss
            return lambda_cls * loss

        return log_loss

    def regression_loss(self, n_anchor: int, huber_delta: float = 1., lambda_reg: float = 1., normalize_w: float = 0.5,
                        epsilon: float = 1e-9):
        """
        :param n_anchor: the number of anchors
        :param huber_delta: ....
        :param lambda_reg: weight value of regression loss function
        :param normalize_w: weight value of normalization
        :param epsilon: ...
        :return: (x_center, y_center, width, height) * n_anchor
                 regression predictions for each of anchors
        """

        def smooth_l1(y_true, y_pred):
            reg_y = y_true[:, :, :, 4 * n_anchor:]

            # cond = tf.equal(reg_y, tf.constant(0.))
            # cls_y = tf.where(cond, tf.zeros_like(reg_y), tf.ones_like(reg_y))
            # cls_y = K.print_tensor(cls_y, 'cls_y')
            # cls_y = tf.Print(cls_y, [cls_y], 'cls_y', first_n=100)

            cls_y = y_true[:, :, :, :4 * n_anchor]

            h1 = K.abs(reg_y - y_pred)
            h2 = K.switch(h1 < huber_delta, 0.5 * h1 ** 2, h1 - 0.5 * huber_delta)
            loss = K.sum(h2) / (normalize_w * K.sum(cls_y) + epsilon)

            self.tensors['reg_y_true'] = y_true
            self.tensors['reg_y_pred'] = y_pred
            self.tensors['reg_reg_y'] = reg_y
            # self.tensors['reg_cond'] = cond
            self.tensors['reg_cls_y'] = cls_y
            # self.tensors['reg_cls_y2'] = cls_y2
            self.tensors['reg_h1'] = h1
            self.tensors['reg_h2'] = h2
            self.tensors['reg_loss'] = loss

            return lambda_reg * loss

        return smooth_l1

    @property
    def model(self) -> Model:
        return self.rpn_model
