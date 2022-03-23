import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models


class ResLayer(tf.keras.Models):
    @staticmethod
    def call(out_previous, out_skip):
        x = layers.Add()([out_previous, out_skip])
        return x

    @staticmethod
    def count_params():
        return 0