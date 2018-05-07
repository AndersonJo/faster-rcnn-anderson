import tensorflow as tf


def test_where():
    x = tf.Variable([0, 0, 0.5, 0.2, 0, 0, 0.6, 1])

    cond = tf.equal(x, tf.constant(0.))
    result = tf.where(cond, tf.zeros_like(x), tf.ones_like(x))
    result = tf.cast(result, tf.int16)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        print()
        print(x.eval())
        print(result.eval())
        print(sess.run(tf.reduce_all(tf.equal(x, x))))
    # assert [0, 0, 0.5, 0.2, 0, 0, 0.6, 1] == x.eval().tolist()
    # assert [0, 0, 1, 1, 0, 0, 1, 1] == result.eval().tolist()
