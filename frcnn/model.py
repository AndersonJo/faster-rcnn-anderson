import math
from typing import List

import tensorflow as tf
from keras import Model, Input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.layers import Conv2D
from keras.optimizers import Adam

from frcnn.rpn import rpn_classification_loss, rpn_regression_loss


class FeatureExtractionNetwork(object):
    def __init__(self, basenet: str = 'vgg16', input_shape: tuple = (None, None, 3)):
        """
        # Base Network (FEN)
        :param basenet: 'vgg16', 'vgg19', 'resnet50'
        :param input_shape: input_shape
        """
        # Initialize input image tensor
        self.input_img = Input(shape=input_shape, name='fen_input_image')

        # Initialize feature extraction network
        self.base_name = basenet.lower().strip()  # name of Base Network (FEN)

        self.model = None
        self.last_tensor = None
        self._init_base_network()

    def _init_base_network(self):
        """
        Set self.model, using pre-trained CNN model
        VGG16, VGG19
        """
        if self.model is not None:
            return self.model

        model = None
        last_tensor = None
        if self.base_name == 'vgg16':
            model = VGG16(include_top=False, input_tensor=self.input_img)
            model = Model(self.input_img, model.layers[-2].output)
            last_tensor = model.layers[-1].output

        elif self.base_name == 'vgg19':
            model = VGG19(include_top=False, input_tensor=self.input_img)
            model = Model(self.input_img, model.layers[-2].output)
            last_tensor = model.layers[-1].output

        elif self.base_name == 'resnet50':
            model = ResNet50(include_top=False, input_tensor=self.input_img)
            model = Model(self.input_img, model.layers[-2].output)
            last_tensor = model.layers[-1].output
        elif self.base_name == 'inception_v3':
            model = InceptionV3(include_top=False, input_tensor=self.input_img)
            last_tensor = model.layers[-1].output

        self.model = model
        self.last_tensor = last_tensor

    @property
    def output(self) -> tf.Tensor:
        return self.last_tensor


class RegionProposalNetwork(object):

    def __init__(self, fen: FeatureExtractionNetwork, anchor_scales: List[int] = (128, 256, 512),
                 anchor_ratios: List[float] = ([1, 1], [1. / math.sqrt(2), 2. / math.sqrt(2)],
                                               [2. / math.sqrt(2), 1. / math.sqrt(2)]),
                 anchor_stride: List[int] = [16, 16], rpn_depth: int = 512):
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

        rpn_losses = [rpn_classification_loss(self.n_anchor), rpn_regression_loss(self.n_anchor)]
        self.rpn_model.compile(Adam(lr=1e-5), loss=rpn_losses)

    @property
    def model(self) -> Model:
        return self.rpn_model
