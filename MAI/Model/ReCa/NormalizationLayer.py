import tensorflow as tf


class Normalization(tf.keras.Model):
    @staticmethod
    def call(inputs):
        x = tf.norm(inputs, name="norm", axis=-1)
        return x
