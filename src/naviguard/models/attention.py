"""naviguard.models.attention — Bahdanau-style additive attention pooling."""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="naviguard")
class AttentionPooling(tf.keras.layers.Layer):
    """Additive attention pooling over an LSTM's per-timestep outputs.

    Learns a scalar importance score per timestep (via a small feed-forward
    scorer), softmaxes those scores across the time axis, and returns the
    weighted sum — a context vector that lets the model attend to whichever
    timesteps in the lookback window matter most for the forecast, instead
    of relying solely on the final LSTM hidden state.
    """

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.score_dense = tf.keras.layers.Dense(units, activation="tanh")
        self.score_out = tf.keras.layers.Dense(1)

    def call(self, h):
        # h: (batch, timesteps, features)
        scores = self.score_out(self.score_dense(h))       # (batch, timesteps, 1)
        weights = tf.nn.softmax(scores, axis=1)             # (batch, timesteps, 1)
        context = tf.reduce_sum(weights * h, axis=1)        # (batch, features)
        return context

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config
