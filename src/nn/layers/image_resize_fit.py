import tensorflow as tf
from tensorflow.image import ResizeMethod


class ResizeImageToFit(tf.keras.layers.Layer):

    def __init__(self, height: int, width: int):
        super().__init__()
        self.width = width
        self.height = height

    @tf.function
    def call(self, inputs):

        if inputs.shape[1] == self.height and inputs.shape[2] == self.width:
            return inputs

        return tf.image.resize_with_pad(
            inputs,
            target_height=self.height,
            target_width=self.width,
            method=ResizeMethod.BILINEAR,
            antialias=True)


    def get_config(self):
        return {
            'height': self.height,
            'width': self.width
        }
