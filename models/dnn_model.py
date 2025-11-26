"""
dnn_model.py

Build a configurable DNN with 3 hidden layers (user requested).
"""
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

def build_dnn_3layer(input_dim, units=(128,64,32), activation='relu', dropout=0.2, lr=1e-3):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(units[0], activation=activation)(inputs)
    if dropout>0:
        x = layers.Dropout(dropout)(x)
    x = layers.Dense(units[1], activation=activation)(x)
    if dropout>0:
        x = layers.Dropout(dropout)(x)
    x = layers.Dense(units[2], activation=activation)(x)
    if dropout>0:
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation='linear')(x)
    model = models.Model(inputs, outputs)
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    return model
