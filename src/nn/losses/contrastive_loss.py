import tensorflow as tf


def contrastive_loss(y_true, y_pred):
    margin = 2.0
    y_true = tf.cast(y_true, float)
    y_pred = tf.cast(y_pred, float)

    y_pred_square = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0.0))
    return tf.reduce_mean((1.0-y_true) * y_pred_square + y_true * margin_square)