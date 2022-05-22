import tensorflow as tf


@tf.keras.utils.register_keras_serializable(name="EuclidianDistanceLayer")
class EuclidianDistance(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(EuclidianDistance, self).__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):

        x = inputs[0]
        y = inputs[1]

        sum = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
        return tf.sqrt(tf.math.maximum(sum, tf.keras.backend.epsilon()))


    def get_config(self):
        return super(EuclidianDistance, self).get_config()
