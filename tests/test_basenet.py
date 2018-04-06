import numpy as np
from keras import Model
from keras import backend as K

from frcnn.net import get_basenet


def test_basenet():
    vgg16 = get_basenet('vgg16')

    # Type check
    assert isinstance(vgg16, Model)

    # Check feature shape
    x = np.random.normal(0, 1, size=(1, 350, 400, 3))
    pred_y = vgg16.predict(x, batch_size=1)
    assert pred_y.shape == (1, 10, 12, 512)

    x = np.random.normal(0, 1, size=(1, 1024, 800, 3))
    pred_y = vgg16.predict(x, batch_size=1)
    assert pred_y.shape == (1, 32, 25, 512)
    K.clear_session()
