"""naviguard.models.lstm_attention — attention-enhanced LSTM model definition."""

import tensorflow as tf

from naviguard.models.attention import AttentionPooling


def build_model(seq_len: int, n_features: int, horizon: int) -> tf.keras.Model:
    """2-layer stacked LSTM + additive attention pooling + multi-step Dense head.

    Same LSTM/Dropout/Dense hyperparameters as the original NaviGuard model
    (64 -> 32 units, 0.2 dropout, Dense(16) compression, Adam lr=1e-3, MSE),
    with two changes: the second LSTM now returns full sequences (so the
    attention layer has something to attend over) and the output layer
    predicts `horizon` steps instead of a single value.
    """
    inputs = tf.keras.Input(shape=(seq_len, n_features))
    x = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LSTM(32, return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = AttentionPooling(32)(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    outputs = tf.keras.layers.Dense(horizon)(x)

    model = tf.keras.Model(inputs, outputs, name="NaviGuard_LSTM_Attention")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model
