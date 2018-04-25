from typing import List

import numpy as np
import tensorflow as tf
from keras import Model, Input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.layers import Conv2D, TimeDistributed, Flatten, Dense, Dropout
from keras.optimizers import Adam

from frcnn.roi import RegionOfInterestPoolingLayer
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

    def __init__(self, fen: FeatureExtractionNetwork, n_anchor: int = 9, rpn_depth: int = 512):
        """
        # Region Proposal Network
        :param n_anchor: the number of anchors (usually 9)
        :param rpn_depth: the depth of the intermediate layer in Region Proposal Network (usually 256 or 512)
        """
        self.fen = fen

        # Initialize Region Proposal Network
        self.n_anchor = n_anchor  # number of anchors
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


class ROINetwork(object):

    def __init__(self, rpn: RegionProposalNetwork, n_class: int = 20,
                 n_roi: int = 32, roi_pool_size: List[int] = (7, 7), roi_method: str = 'resize'):
        """
        # Region of Interest
        :param n_class: the number of classes (PASCAL VOC is 20)
        :param n_roi: the number of regions of interest
        :param roi_pool_size: size of roi pooling layer. (height, width)
        :param roi_method: ('resize', 'pooling') how to do region of interest
            - resize: this method is very simple but works properly
            - pooling: slow but it does what paper says to do exactly
        """
        self.rpn = rpn

        # Initiliaze Region of Interest
        assert len(roi_pool_size) == 2
        self.roi_input = Input(shape=(n_roi, 4))
        self.n_class = n_class  # number of classes (like car, human, bike, etc..)
        self.n_roi = n_roi
        self.roi_method = roi_method
        self.pool_height = roi_pool_size[0]
        self.pool_widht = roi_pool_size[1]
        self.roi_cls_output = None
        self.roi_reg_output = None
        self.roi_pooling_layer = RegionOfInterestPoolingLayer(size=roi_pool_size, n_roi=n_roi, method=roi_method)
        self._init_classifier()

    def _init_classifier(self) -> List[np.ndarray]:
        roi_pooled_output = self.roi_pooling_layer([self.rpn.fen.output, self.roi_input])

        h = TimeDistributed(Flatten(name='flatten'))(roi_pooled_output)
        h = TimeDistributed(Dense(4096, activation='relu', name='roi_fc1'))(h)
        h = TimeDistributed(Dropout(0.5))(h)
        h = TimeDistributed(Dense(4096, activation='relu', name='roi_fc2'))(h)
        h = TimeDistributed(Dropout(0.5))(h)

        cls_output = TimeDistributed(Dense(self.n_class, activation='softmax', kernel_initializer='zero'),
                                     name='roi_class_{}'.format(self.n_class))(h)

        reg_output = TimeDistributed(Dense(4 * (self.n_class - 1), activation='linear', kernel_initializer='zero'),
                                     name='roi_regress_{}'.format(self.n_class))(h)

        self.roi_cls_output = cls_output
        self.roi_reg_output = reg_output

        return [cls_output, reg_output]
