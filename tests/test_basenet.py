import numpy as np
from keras import Model
from keras import backend as K

from frcnn.model import FasterRCNN


def test_faster_rcnn_model():
    def _test_faster_rcnn_model(base_model, width: int, height: int):
        frcnn = FasterRCNN(base_model, rpn_depth=512)
        assert isinstance(frcnn.base_model, Model)

        # Check Output Shape
        x = np.random.normal(0, 1, size=(1, width, height, 3))
        pred_y = frcnn.base_model.predict(x, batch_size=1)
        assert (1, *frcnn.get_output_size(width, height)) == pred_y.shape, 'base:{0} width:{1}, height:{2}'.format(
            frcnn.base_name, width, height)

        # Check RPN Model
        x = np.random.normal(0, 1, size=(1, 350, 400, 3))
        output_cls, output_reg = frcnn.rpn_model.predict(x, batch_size=1)

    _width = np.random.randint(200, 5000)
    _height = np.random.randint(200, 5000)
    _test_faster_rcnn_model('vgg16', _width, _height)
    _test_faster_rcnn_model('vgg19', _width, _height)

    # TODO: Add resnet50 and inception_v3
    # _test_faster_rcnn_model('resnet50', _width, _height)
    # _test_faster_rcnn_model('inception_v3', _width, _height)

    K.clear_session()
