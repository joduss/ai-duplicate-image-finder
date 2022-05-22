from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.python.data import AUTOTUNE

from src.data.dataset.dataset import Dataset
from src.data.dataset.image_shape import ImageShape


class TfDatasetTransformer():
    """
    Images are not processed by default.
    they may be rescaled in [-1;0] interval if rescale=True
    """

    def __init__(self, image_shape: ImageShape, rescale=False, resize=True):
        self.image_shape = image_shape
        self.rescale = rescale
        self.resize = resize


    def transform_to_tf_dataset(self, dataset: Dataset,
                                shuffle_buffer_size=None, batch_size=64, cache_path=None, shuffle: bool = False) -> tf.data.Dataset:
        pair_tuples = list(map(lambda pair: (pair.image_a, pair.image_b, int(pair.similar)), dataset.image_pairs))
        pair_array = np.array(pair_tuples)

        tf_dataset = tf.data.Dataset.from_tensor_slices(
            (pair_array[:, 0], pair_array[:, 1], pair_array[:, 2].astype(int))
        )

        tf_dataset = tf_dataset.map(self._parse, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

        # if cache_path is not None:
        #     tf_dataset = tf_dataset.cache(cache_path)

        if shuffle == True and shuffle_buffer_size is None:
            shuffle_buffer_size = 32

        if shuffle:
            tf_dataset = tf_dataset.shuffle(buffer_size=shuffle_buffer_size)

        # tf_dataset = tf_dataset.map(self.print_loaded_image_name_tf)

        if batch_size is not None:
            tf_dataset = tf_dataset.batch(batch_size)

        return tf_dataset.prefetch(AUTOTUNE)


    def print_loaded_image_name_tf(self, image_a_path: tf.Tensor, image_b_path: tf.Tensor, similar: tf.Tensor):
        return tf.py_function(self.print_loaded_image_name, [image_a_path, image_b_path, similar],
                              Tout=[image_a_path.dtype, image_a_path.dtype, similar.dtype])


    def print_loaded_image_name(self, image_a_path: tf.Tensor, image_b_path: tf.Tensor, similar: tf.Tensor):
        print(image_a_path, end='\r')
        return image_a_path, image_b_path, similar


    @tf.function
    def _parse(self, image_a_path: tf.Tensor, image_b_path: tf.Tensor, similar: tf.Tensor):
        image_a = self._load_image(image_a_path)
        image_b = self._load_image(image_b_path)

        return (image_a, image_b), similar


    def pr(self, a):
        print(a)
        return a


    @tf.function
    def _load_image(self, image_path: tf.Tensor) -> object:
        raw = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(raw)

        if self.rescale:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = tf.multiply(image, 2)
            image = tf.subtract(image, 1)

        if self.resize:
            image = tf.image.resize_with_pad(image,
                                             target_height=self.image_shape.height,
                                             target_width=self.image_shape.width)
        if image.shape[2] == 1:
            image = tf.repeat(image, repeats=3, axis=2)

        return image
