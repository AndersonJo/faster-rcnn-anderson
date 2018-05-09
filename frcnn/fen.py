import tensorflow as tf
from keras import Model, Input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19

from frcnn.config import Config


class FeatureExtractionNetwork(object):
    def __init__(self, config: Config, input_shape: tuple = (None, None, 3)):
        """
        # Base Network (FEN)
        :param basenet: 'vgg16', 'vgg19', 'resnet50'
        :param input_shape: input_shape
        """
        # Initialize input image tensor
        self.image_input = Input(shape=input_shape, name='fen_input_image')

        # Initialize feature extraction network
        self.base_name = config.net_name.lower().strip()  # name of Base Network (FEN)

        self.model = None
        self.last_tensor = None
        self._init_fen_network()

    def _init_fen_network(self):
        """
        Set self.model, using pre-trained CNN model
        VGG16, VGG19
        """
        if self.model is not None:
            return self.model

        model = None
        last_tensor = None
        if self.base_name == 'vgg16':
            model = VGG16(include_top=False, input_tensor=self.image_input)
            model = Model(self.image_input, model.layers[-2].output)
            last_tensor = model.layers[-1].output

        elif self.base_name == 'vgg19':
            model = VGG19(include_top=False, input_tensor=self.image_input)
            model = Model(self.image_input, model.layers[-2].output)
            last_tensor = model.layers[-1].output

        elif self.base_name == 'resnet50':
            model = ResNet50(include_top=False, input_tensor=self.image_input)
            model = Model(self.image_input, model.layers[-2].output)
            last_tensor = model.layers[-1].output
        elif self.base_name == 'inception_v3':
            model = InceptionV3(include_top=False, input_tensor=self.image_input)
            last_tensor = model.layers[-1].output

        self.model = model
        self.last_tensor = last_tensor

    @property
    def output(self) -> tf.Tensor:
        return self.last_tensor
