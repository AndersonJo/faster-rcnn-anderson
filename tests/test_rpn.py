import keras.backend as K
import numpy as np
import tensorflow as tf


def test_binary_cross_entropy():
    y_true = np.array([0, 0, 0, 1, 1], dtype=np.float64)
    y_pred = np.array([0, 0, 0.5, 0.9, 1], dtype=np.float64)

    _y_true = tf.placeholder('float')
    _y_pred = tf.placeholder('float')
    result = _y_true * K.binary_crossentropy(_y_true, _y_pred)

    with K.get_session() as sess:
        print(sess.run(result, feed_dict={_y_true: y_true, _y_pred: y_pred}))
