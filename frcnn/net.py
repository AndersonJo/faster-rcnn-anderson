from keras import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50


def get_basenet(net: str = 'vgg16', input_shape=(None, None, 3)) -> Model:
    if hasattr(get_basenet, 'model') and get_basenet.model is not None:
        return get_basenet.model

    model = None
    if net == 'vgg16':
        model = VGG16(include_top=False, input_shape=input_shape)
    elif net == 'vgg19':
        model = VGG19(include_top=False, input_shape=input_shape)
    elif net == 'resnet50':
        model = ResNet50(include_top=False, input_shape=input_shape)
    get_basenet.model = model
    return model
