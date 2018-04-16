from typing import Tuple

from keras import Model, Input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.layers import Conv2D


class FasterRCNN(object):

    def __init__(self, basenet: str = 'vgg16', n_anchor: int = 9, input_shape: tuple = (None, None, 3),
                 rpn_depth: int = 512):
        """
        :param basenet: 'vgg16', 'vgg19', 'resnet50'
        :param n_anchor: the number of anchors (usually 9)
        :param input_shape: input_shape
        :param rpn_depth: the depth of the intermediate layer in Region Proposal Network (usually 256 or 512)
        """
        # Set instance members
        self.base_name = basenet.lower().strip()
        self.input_shape = input_shape
        self.n_anchor = n_anchor

        # Initialize Input Tensor
        self.input_img = None
        self._init_input()

        # Initialize Base Network (shared network)
        self.base_model = None
        self.base_last_tensor = None
        self._init_base_network()

        # Initialize Region Proposal Network
        self.rpn_layer = None
        self.rpn_cls = None
        self.rpn_reg = None
        self.rpn_model = None
        self._init_rpn(rpn_depth)

    def _init_input(self):
        self.input_img = Input(shape=self.input_shape, name='image_input')

    def _init_base_network(self):
        """
        Set self.base_model, using pre-trained CNN model
        VGG16, VGG19
        """
        if self.base_model is not None:
            return self.base_model

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

        self.base_model = model
        self.base_last_tensor = last_tensor

    def _init_rpn(self, rpn_depth: int):
        """
        This method initializes Region Proposal Network after base network (i.e. VGG16)
        :param rpn_depth:
        :return:
        """
        # intermediate layer assures that specific depth like 256, 512 is
        intermediate_layer = Conv2D(rpn_depth, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                    activation='relu', kernel_initializer='glorot_uniform',
                                    name='rpn_intermediat_layer')(self.base_last_tensor)

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
        self.rpn_model = Model(self.input_img, outputs=[self.rpn_cls, self.rpn_reg])
        # self.rpn_model.compile(Adam(lr=1e-5), loss=[])

    @staticmethod
    def rpn_cls_loss(y_true, y_pred, rpn_lambda: int = 10):

        # rpn_lambda * K.sum(y_true[:, :, :, :num_anchors] * K.binary_crossentropy(y_pred[:, :, :, :],
        #                                                                          y_true[:, :, :,
        #                                                                          num_anchors:])) / K.sum(
        #     epsilon + y_true[:, :, :, :num_anchors])
        pass

    @staticmethod
    def rpn_reg_loss():
        pass


def singleton_frcnn(*args, **kwargs) -> FasterRCNN:
    """
    You are advised to use this method in production environment.
    :return an instance of FasterRCNN class.
    """
    if hasattr(singleton_frcnn, 'singleton') and singleton_frcnn.singleton is not None:
        return singleton_frcnn.singleton

    singleton_frcnn.singleton = FasterRCNN(*args, **kwargs)
    return singleton_frcnn.singleton
