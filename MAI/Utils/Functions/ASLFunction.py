import numpy as np  # linear algebra
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import sigmoid
from tensorflow.python.keras.backend import mean, exp, log, sum, min, max
from tensorflow_addons.utils.types import FloatTensorLike


@tf.function()
def AsymetricLossOptimized(y_true, y_pred):
    gamma_neg = 5
    gamma_pos = 0
    clip = 0.01

    y_true = tf.convert_to_tensor(y_true, np.float32)
    y_pred = tf.convert_to_tensor(y_pred, np.float32)

    targets = y_true

    xs_pos = sigmoid(y_pred)
    xs_neg = 1.0 - xs_pos

    if clip is not None and clip > 0:
        xs_neg = tf.clip_by_value(
            tf.math.add(
                xs_neg,
                clip
            ),
            clip_value_min=min(
                tf.math.add(
                    xs_neg,
                    clip
                )
            ),
            clip_value_max=1
        )

    loss = focal_loss(y_true, y_pred)

    if gamma_neg > 0 or gamma_pos > 0:
        pt0 = xs_pos * targets
        pt1 = xs_neg * (1 - targets)
        pt = pt0 + pt1
        one_sided_gamma = gamma_pos * targets + gamma_neg * (1 - targets)
        one_sided_w = tf.math.pow(1 - pt, one_sided_gamma)
        loss *= one_sided_w

    return mean(loss)


def focal_loss(y_true, y_pred, alpha: FloatTensorLike = 0.25,
               gamma: FloatTensorLike = 2.0,
               from_logits: bool = False):
    if gamma and gamma < 0:
        raise ValueError("Value of gamma should be greater than or equal to zero.")

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, dtype=y_pred.dtype)

    # Get the cross_entropy for each entry
    ce = K.binary_crossentropy(y_true, y_pred, from_logits=from_logits)

    # If logits are provided then convert the predictions into probabilities
    if from_logits:
        pred_prob = tf.sigmoid(y_pred)
    else:
        pred_prob = y_pred

    p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
    alpha_factor = 1.0
    modulating_factor = 1.0

    if alpha:
        alpha = tf.cast(alpha, dtype=y_true.dtype)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

    if gamma:
        gamma = tf.cast(gamma, dtype=y_true.dtype)
        modulating_factor = tf.pow((1.0 - p_t), gamma)

    # compute the final loss and return
    return alpha_factor * modulating_factor * ce
