"""
Evaluation script for trained DQN agent
"""
import numpy as np
import tensorflow as tf
from agents.dqn_agent import DQNAgent
from utils.load_data import load_data
from environ.vppenv import VPPEnv
from CONFIG import NUM_EVS, ACTION_PER_PILE


def evaluate_agent(agent, episodes=5, max_steps=1000):
    """
    Evaluate trained agent
    
    Args:
        agent: Trained DQN agent
        episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
    """
    print("Loading data...")
    data = load_data()
    
    print("Creating environment...")
    env = VPPEnv(data)
    
    episode_rewards = []
    
    print(f"\nEvaluating agent for {episodes} episodes...\n")
    
    for episode in range(episodes):
        obs = env.reset()
        if obs is None:
            obs = np.zeros(11, dtype=np.float32)
        
        episode_reward = 0.0
        done = False
        step = 0
        
        while not done and step < max_steps:
            # Select greedy action (no exploration)
            actions = agent.select_action(obs, training=False)
            
            # Convert action indices to one-hot
            action_onehot = np.zeros(NUM_EVS * ACTION_PER_PILE)
            for ev_idx, action_idx in enumerate(actions):
                action_onehot[ev_idx * ACTION_PER_PILE + action_idx] = 1
            
            reward, next_obs = env.step(action_onehot)
            if next_obs is None:
                next_obs = np.zeros(11, dtype=np.float32)
            
            episode_reward += reward
            obs = next_obs
            step += 1
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Total Reward = {episode_reward:.2f}")
    
    avg_reward = np.mean(episode_rewards)
    print(f"\nAverage Reward: {avg_reward:.2f}")
    print(f"Std Dev: {np.std(episode_rewards):.2f}")
    
    return episode_rewards


if __name__ == "__main__":
    print("Creating agent...")
    agent = DQNAgent(
        obs_dim=11,
        num_evs=NUM_EVS,
        num_actions=ACTION_PER_PILE,
        hidden_dims=[64, 128, 64],
        learning_rate=1e-3,
        gamma=0.99
    )
    
    # Load pre-trained weights if available
    try:
        print("Loading pre-trained model...")
        agent.q_network.model.load_weights("model/dqn_agent.h5")
        print("Model loaded successfully!")
    except:
        print("No pre-trained model found. Using random initialization.")
    
    # Evaluate
    rewards = evaluate_agent(agent, episodes=5, max_steps=1000)
