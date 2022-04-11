import tensorflow as tf



class EuclidianDistance(tf.keras.layers.Layer):

    @tf.function
    def call(self, inputs):

        x = inputs[0]
        y = inputs[1]

        sum = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
        return tf.sqrt(tf.math.maximum(sum, tf.keras.backend.epsilon()))
