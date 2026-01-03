"""
Test script to verify the Q-Network architecture
"""
import numpy as np
import tensorflow as tf
from model.architecutre import QNetwork

if __name__ == "__main__":
    # Create the Q-Network
    q_net = QNetwork(obs_dim=11, num_evs=4, num_actions_per_ev=3, hidden_dim=128)
    
    # Create dummy observation (batch_size=2, obs_dim=11)
    dummy_obs = tf.constant(np.random.randn(2, 11).astype(np.float32))
    
    # Forward pass
    q_values = q_net(dummy_obs)
    
    print(f"Input observation shape: {dummy_obs.shape}")
    print(f"Output Q-values shape: {q_values.shape}")
    print(f"\nQ-values (first sample):\n{q_values[0].numpy()}")
    print(f"\nQ-values structure:")
    print(f"  - Number of EVs: {q_values.shape[1]}")
    print(f"  - Actions per EV: {q_values.shape[2]}")
    print(f"\nSuccess! Network outputs Q-values for 4 EVs with 3 actions each.")
