"""
Deep Q-Network (DQN) Training Algorithm for VPP Environment
Implements base Q-learning with experience replay and target network
"""
import numpy as np
import tensorflow as tf
from collections import deque
from model.architecutre import QNetwork
from utils.load_data import load_data
from environ.vppenv import VPPEnv
from CONFIG import NUM_EVS, ACTION_PER_PILE, LAMBDA1


class ReplayBuffer:
    """Experience replay buffer for DQN training"""
    
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        """Add experience to replay buffer"""
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """Sample random batch from replay buffer"""
        batch = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        experiences = [self.buffer[i] for i in batch]
        
        states = np.array([exp[0] for exp in experiences])
        actions = np.array([exp[1] for exp in experiences])
        rewards = np.array([exp[2] for exp in experiences])
        next_states = np.array([exp[3] for exp in experiences])
        dones = np.array([exp[4] for exp in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Network Agent for VPP control"""
    
    def __init__(self, obs_dim=11, num_evs=NUM_EVS, num_actions=ACTION_PER_PILE,
                 hidden_dims=[64, 128, 64], learning_rate=1e-3, gamma=0.99):
        """
        Initialize DQN Agent
        
        Args:
            obs_dim: Observation dimension
            num_evs: Number of EVs
            num_actions: Actions per EV
            hidden_dims: List of hidden layer dimensions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
        """
        self.obs_dim = obs_dim
        self.num_evs = num_evs
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = 0.7  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.95
        
        # Q-Networks (online and target)
        self.q_network = QNetwork(obs_dim, num_evs, num_actions, hidden_dims)
        self.target_network = QNetwork(obs_dim, num_evs, num_actions, hidden_dims)
        
        # Copy weights from online to target network
        self.update_target_network()
        
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(max_size=100000)
        
        # Training stats
        self.loss_history = []
    
    def update_target_network(self):
        """Copy weights from online network to target network"""
        self.target_network.model.set_weights(self.q_network.model.get_weights())
    
    def select_action(self, obs, training=True):
        """
        Select action using epsilon-greedy policy
        
        Args:
            obs: Observation
            training: Whether in training mode
            
        Returns:
            Action indices for each EV
        """
        if training and np.random.random() < self.epsilon:
            # Random action
            actions = np.random.randint(0, self.num_actions, size=self.num_evs)
        else:
            # Greedy action
            obs_tensor = tf.constant(obs[np.newaxis, :], dtype=tf.float32)
            q_values = self.q_network.model(obs_tensor)  # List of (1, num_actions) for each EV
            actions = np.array([tf.argmax(q_val[0]).numpy() for q_val in q_values])
        
        return actions
    
    def remember(self, obs, actions, reward, next_obs, done):
        """Store experience in replay buffer"""
        self.replay_buffer.add((obs, actions, reward, next_obs, done))
    
    def train(self, batch_size=32):
        """
        Train the Q-network on a batch from replay buffer
        
        Args:
            batch_size: Size of training batch
            
        Returns:
            Loss value
        """
        if len(self.replay_buffer) < batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Convert to TensorFlow tensors
        states = tf.constant(states, dtype=tf.float32)
        actions = tf.constant(actions, dtype=tf.int32)
        rewards = tf.constant(rewards, dtype=tf.float32)
        next_states = tf.constant(next_states, dtype=tf.float32)
        dones = tf.constant(dones, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            # Current Q-values from online network
            q_values = self.q_network.model(states)  # List of (batch_size, num_actions)
            
            # Target Q-values from target network
            target_q_values = self.target_network.model(next_states)  # List of (batch_size, num_actions)
            
            # Build targets for each EV
            targets = []
            for ev_idx in range(self.num_evs):
                # Max Q-value for next state: (batch_size,)
                max_next_q = tf.reduce_max(target_q_values[ev_idx], axis=1)
                
                # Q-learning target: r + gamma * max(Q(s', a')) * (1 - done)
                target_q = rewards + self.gamma * max_next_q * (1.0 - dones)
                targets.append(target_q)
            
            # Compute loss
            loss = 0.0
            for ev_idx in range(self.num_evs):
                # Get batch indices
                batch_indices = tf.range(batch_size)
                action_indices = actions[:, ev_idx]
                
                # Select Q-values for actions taken: (batch_size,)
                selected_q = tf.gather_nd(
                    q_values[ev_idx],
                    tf.stack([batch_indices, action_indices], axis=1)
                )
                
                # MSE loss: E[(Q(s, a) - target)^2]
                loss += tf.reduce_mean(tf.square(selected_q - targets[ev_idx]))
                
                # Regularization term
                # q_values[ev_idx] shape is 128x3 and selected_q shape is 128
                # reduce each element in selected_q from each row in q_values[ev_idx]
                diff = q_values[ev_idx] - tf.expand_dims(selected_q, axis=1)

                # diff = q_values[ev_idx] - selected_q
                reg_loss = tf.reduce_max(tf.abs(diff**2))
                loss += LAMBDA1 * reg_loss  
                
            # Average loss across EVs
            loss /= self.num_evs
        # Backpropagation
        gradients = tape.gradient(loss, self.q_network.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.model.trainable_variables))
        
        self.loss_history.append(float(loss))
        return float(loss)

    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_dqn(episodes=100, batch_size=32, update_target_freq=10):
    """
    Main training loop for DQN agent
    
    Args:
        episodes: Number of training episodes
        batch_size: Batch size for training
        update_target_freq: Frequency of target network updates
    """
    print("Loading data...")
    data = load_data()
    
    print("Creating environment...")
    env = VPPEnv(data)
    
    print("Initializing DQN Agent...")
    agent = DQNAgent(
        obs_dim=11,
        num_evs=NUM_EVS,
        num_actions=ACTION_PER_PILE,
        hidden_dims=[64, 128, 64],
        learning_rate=1e-3,
        gamma=0.99
    )
    
    episode_rewards = []
    
    print(f"\nStarting training for {episodes} episodes...\n")
    
    for episode in range(episodes):
        obs = env.reset()
        if obs is None:
            obs = np.zeros(11, dtype=np.float32)
        
        episode_reward = 0.0
        done = False
        step = 0
        max_steps = 1000
        
        while not done and step < max_steps:
            # Select and execute action
            actions = agent.select_action(obs, training=True)
            
            # Convert action indices to one-hot for environment
            action_onehot = np.zeros(NUM_EVS * ACTION_PER_PILE)
            for ev_idx, action_idx in enumerate(actions):
                action_onehot[ev_idx * ACTION_PER_PILE + action_idx] = 1
            
            reward, next_obs = env.step(action_onehot)
            if next_obs is None:
                next_obs = np.zeros(11, dtype=np.float32)
            
            episode_reward += reward
            
            # Store in replay buffer
            agent.remember(obs, actions, reward, next_obs, done)
            
            # Train on batch
            loss = agent.train(batch_size)
            
            obs = next_obs
            step += 1
        
        # Update target network
        if (episode + 1) % update_target_freq == 0:
            agent.update_target_network()
        
        # Decay exploration rate
        agent.decay_epsilon()
        
        episode_rewards.append(episode_reward)
        
        # Logging
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}/{episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Avg Reward (10): {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    print("\nTraining completed!")
    return agent, episode_rewards


if __name__ == "__main__":
    agent, rewards = train_dqn(episodes=50, batch_size=32, update_target_freq=10)
    
    # Save model
    print("\nSaving model...")
    agent.q_network.model.save("model/dqn_agent.h5")
    print("Model saved to model/dqn_agent.h5")
