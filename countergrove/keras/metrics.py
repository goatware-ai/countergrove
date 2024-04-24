"""Module for metrics from countergrove predictions."""

import tensorflow as tf
from countergrove import utils


@tf.keras.utils.register_keras_serializable(package="countergrove")
def propensity_score_crossentropy(y_true: tf.Tensor, y_pred: tf.Tensor):
    """Computes cross entropy for the propensity score. Used as a metric in countergrove model."""
    _, _, _, propensity_score = utils.split_y_pred(y_pred)
    treatment_true = y_true[:, 0:1]
    return tf.keras.metrics.binary_crossentropy(
        y_true=treatment_true, y_pred=propensity_score
    )


@tf.keras.utils.register_keras_serializable(package="countergrove")
class PropensityScoreBinaryCrossentropy(tf.keras.metrics.BinaryCrossentropy):
    """Computes cross entropy for the propensity score. Used as a metric in countergrove model."""

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Updates state."""
        _, _, _, propensity_score = utils.split_y_pred(y_pred)
        treatment_true = y_true[:, 0:1]
        super().update_state(
            y_true=treatment_true, y_pred=propensity_score, sample_weight=sample_weight
        )


@tf.keras.utils.register_keras_serializable(package="countergrove")
class PropensityScoreAUC(tf.keras.metrics.AUC):
    """AUC computed on the ouptut for propensity part."""

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Updates state"""
        _, _, _, propensity_score = utils.split_y_pred(y_pred)
        treatment_true = y_true[:, 0:1]
        super().update_state(
            y_true=treatment_true, y_pred=propensity_score, sample_weight=sample_weight
        )
