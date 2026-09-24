"""naviguard.models.lstm_attention — attention-enhanced LSTM model definition."""

import tensorflow as tf

from naviguard.config import TARGET_IDX, ModelHParams
from naviguard.models.attention import AttentionPooling, LastValueSkip


def build_model(
    seq_len: int, n_features: int, horizon: int, hparams: ModelHParams | None = None
) -> tf.keras.Model:
    """Stacked LSTM + additive attention pooling + multi-step Dense head.

    Defaults (64 -> 32 units, 0.2 dropout, Dense(16), Adam lr=1e-3, MSE) come
    from ModelHParams. The second LSTM returns full sequences so the
    attention layer has something to attend over, and the output layer
    predicts `horizon` steps at once.
    """
    hp = hparams or ModelHParams()
    inputs = tf.keras.Input(shape=(seq_len, n_features))
    x = inputs
    for units in hp.lstm_units:
        x = tf.keras.layers.LSTM(units, return_sequences=True)(x)
        x = tf.keras.layers.Dropout(hp.dropout)(x)
    x = AttentionPooling(hp.lstm_units[-1])(x)
    x = tf.keras.layers.Dense(hp.dense_units, activation="relu")(x)
    outputs = tf.keras.layers.Dense(horizon)(x)
    if hp.residual_skip:
        outputs = LastValueSkip(TARGET_IDX)([outputs, inputs])

    model = tf.keras.Model(inputs, outputs, name="NaviGuard_LSTM_Attention")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.learning_rate), loss="mse")
    return model
