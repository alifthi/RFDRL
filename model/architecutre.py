import tensorflow as tf
from tensorflow.keras import layers, regularizers
import numpy as np
from CONFIG import LAMBDA2

class QNetwork:

    def __init__(self, obs_dim=11, num_evs=4, num_actions_per_ev=3, hidden_dim=[64,128,64]):        
        self.obs_dim = obs_dim
        self.num_evs = num_evs
        self.num_actions_per_ev = num_actions_per_ev
        self.total_output = num_evs * num_actions_per_ev
        self.hidden_dim = hidden_dim
        self.model = None
        self.build_model()
        
    def build_model(self):
        inputs = tf.keras.Input(shape=(self.obs_dim,))
        
        # Add input normalization to stabilize training
        x = layers.Normalization(axis=-1)(inputs)
        
        # Hidden layers with layer normalization and L2 regularization
        x = layers.Dense(self.hidden_dim[0], activation='relu', 
                        kernel_regularizer=regularizers.l2(LAMBDA2))(x)
        x = layers.LayerNormalization()(x)
        
        for i in range(1, len(self.hidden_dim)):
            x = layers.Dense(self.hidden_dim[i], activation='relu',
                            kernel_regularizer=regularizers.l2(LAMBDA2))(x)
            x = layers.LayerNormalization()(x)
        
        outputs = []
        for _ in range(self.num_evs):
            # Output layer with scaled initialization to prevent divergence
            ev_output = layers.Dense(self.num_actions_per_ev, 
                                    activation='linear',
                                    kernel_initializer=tf.keras.initializers.RandomUniform(minval=-3e-3, maxval=3e-3),
                                    kernel_regularizer=regularizers.l2(LAMBDA2))(x)
            outputs.append(ev_output)
        
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        
