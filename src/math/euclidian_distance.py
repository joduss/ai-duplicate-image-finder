import tensorflow as tf


def euclidian_distance(vectors):
    x, y = vectors
    sum = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(tf.math.maximum(sum, 0))