import tensorflow as tf
from tensorflow.keras import layers, models

def build_autoencoder(input_dim: int) -> tf.keras.Model:
    """
    Simple fully-connected autoencoder for 1D beats.
    Input: vector of length input_dim
    """
    inputs = layers.Input(shape=(input_dim,))

    # Encoder
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dense(64, activation="relu")(x)
    encoded = layers.Dense(32, activation="relu", name="latent_vector")(x)

    # Decoder
    x = layers.Dense(64, activation="relu")(encoded)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(input_dim, activation="linear")(x)

    model = models.Model(inputs, outputs, name="ecg_autoencoder")
    return model
