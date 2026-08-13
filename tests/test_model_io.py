"""Tests for the attention-LSTM model definition and save/load roundtrip.

The save/load roundtrip test in particular guards against custom-layer
serialization regressions (AttentionPooling once had a manual build()
override that broke Keras's deserialization with a list/tuple mismatch —
this test would have caught that).
"""

import numpy as np
import tensorflow as tf

from naviguard.models.attention import AttentionPooling
from naviguard.models.lstm_attention import build_model


def test_build_model_output_shape():
    seq_len, n_features, horizon = 8, 3, 4
    model = build_model(seq_len, n_features, horizon)

    X = np.random.default_rng(0).normal(size=(5, seq_len, n_features)).astype("float32")
    y_pred = model.predict(X, verbose=0)

    assert y_pred.shape == (5, horizon)


def test_model_save_load_roundtrip(tmp_path):
    seq_len, n_features, horizon = 6, 3, 2
    model = build_model(seq_len, n_features, horizon)

    X = np.random.default_rng(1).normal(size=(4, seq_len, n_features)).astype("float32")
    y = np.random.default_rng(2).normal(size=(4, horizon)).astype("float32")
    model.fit(X, y, epochs=1, verbose=0)  # forces layers to build with real weights

    model_path = str(tmp_path / "model.keras")
    model.save(model_path)

    reloaded = tf.keras.models.load_model(model_path, custom_objects={"AttentionPooling": AttentionPooling})
    y_pred_original = model.predict(X, verbose=0)
    y_pred_reloaded = reloaded.predict(X, verbose=0)

    assert np.allclose(y_pred_original, y_pred_reloaded, atol=1e-5)
