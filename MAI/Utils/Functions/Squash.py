import tensorflow as tf


def squash(vectors, axis=-1):
    epsilon = 1e-8
    vector_square_norm = tf.math.reduce_sum(tf.math.square(vectors), axis=axis, keepdims=True) + epsilon
    return (vector_square_norm / (1 + vector_square_norm)) * \
           (vectors / tf.math.sqrt(vector_square_norm)) + epsilon
