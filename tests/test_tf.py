import tensorflow as tf
import numpy as np


def test_where():
    x = tf.Variable([0, 0, 0.5, 0.2, 0, 0, 0.6, 1], dtype=tf.float64)

    cond = tf.equal(x, tf.constant(0., dtype=tf.float64))
    result = tf.where(cond, tf.zeros_like(x), tf.ones_like(x))
    result = tf.cast(result, tf.int16)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        assert [0., 0., 0.5, 0.2, 0., 0., 0.6, 1.] == x.eval().tolist()
        assert [0, 0, 1, 1, 0, 0, 1, 1] == result.eval().tolist()


def test_dynamic_tensorflow():
    tf.enable_eager_execution()
    x = np.array([[2., 4.]])
    print(tf.matmul(x, x.T))
