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

    def build(self, input_shape):
        input_shape = tuple(input_shape)
        self.score_dense.build(input_shape)
        self.score_out.build(input_shape[:-1] + (self.units,))
        super().build(input_shape)

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


@tf.keras.utils.register_keras_serializable(package="naviguard")
class LastValueSkip(tf.keras.layers.Layer):
    """Adds the window's last observed target value to every predicted step.

    The network then only learns the *change* from the last observation
    (a persistence forecast is the starting point), which makes it robust to
    slow level wander that pushes test values outside the training range.
    """

    def __init__(self, target_idx: int, **kwargs):
        super().__init__(**kwargs)
        self.target_idx = target_idx

    def call(self, inputs):
        delta, window = inputs                      # (batch, horizon), (batch, timesteps, features)
        return delta + window[:, -1, self.target_idx][:, None]

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config.update({"target_idx": self.target_idx})
        return config
