from utils.load_data import load_data
from environ.vppenv import VPPEnv
from agents.dqn_agent import DQNAgent
from aggregator.aggregator import aggregator
from CONFIG import EPISODES_PER_AGGREGATION, NUM_EVS, ACTION_PER_PILE, MODEL_SAVE_PATH, AGENTS_SAVE_PATH
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    print("="*60)
    print("DQN Agent Training for VPP Environment")
    print("="*60)
    
    # Load data
    print("\n[1/4] Loading data...")
    data = load_data()
    print(f"✓ Data loaded successfully")
    
    # Create environment
    print("\n[2/4] Creating environment...")
    env = VPPEnv(data)
    print(f"✓ Environment created")
    print(f"  - Observation space: {env.observation_space.shape}")
    print(f"  - Number of EVs: {NUM_EVS}")
    print(f"  - Actions per EV: {ACTION_PER_PILE}")
    
    # Initialize agent
    print("\n[3/4] Initializing DQN Agent...")
    aggregator_instance = aggregator()
    print(f"✓ Agent initialized")
    print(f"  - Network architecture: {aggregator_instance.main_model.q_network.hidden_dim}")
    print(f"  - Initial epsilon: {aggregator_instance.main_model.epsilon}")
    print(f"  - Replay buffer size: {aggregator_instance.main_model.replay_buffer.buffer.maxlen}")
    
    # Load model if exists
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"\n✓ Loading pre-trained model from {MODEL_SAVE_PATH}...")
        aggregator_instance.main_model.q_network.model.load_weights(MODEL_SAVE_PATH)
        aggregator_instance.set_agents_weights(aggregator_instance.main_model.q_network.model.get_weights())
        print(f"✓ Model loaded successfully")
    else:
        print(f"\n! No pre-trained model found at {MODEL_SAVE_PATH}")
        print(f"! Starting from scratch...")
    
    # Training
    print("\n[4/4] Training Agent...")
    print("-" * 60)
    
    episodes = 200
    batch_size = 128
    update_target_freq = 10
    episode_rewards_list = [[] for _ in range(len(aggregator_instance.agents))]
    max_steps = 35050
    
    for episode in range(episodes):
        for agent_idx, agent in enumerate(aggregator_instance.agents):
            episode_rewards = episode_rewards_list[agent_idx]
            obs = env.reset()
            if obs is None:
                obs = np.zeros(11, dtype=np.float32)
                
            done = False
            episode_reward = 0.0
            step = 0
            print(f"\nEpisode {episode + 1}/{episodes} Agent {agent_idx + 1}/{len(aggregator_instance.agents)}")
            while not done and step < max_steps:
                # Select action using epsilon-greedy policy
                actions = agent.select_action(obs, training=True)
                # Convert action indices to one-hot for environment
                action_onehot = np.zeros(NUM_EVS * ACTION_PER_PILE)
                for ev_idx, action_idx in enumerate(actions):
                    action_onehot[ev_idx * ACTION_PER_PILE + action_idx] = 1
                
                # Step environment
                next_obs, reward, done,_ = env.step(action_onehot)
                # reward = -reward
                if next_obs is None:
                    next_obs = np.zeros(11, dtype=np.float32)
                
                episode_reward += reward
                
                # Store experience in replay buffer
                agent.remember(obs, actions, reward, next_obs, done)
                
                # Train on batch
                if step % 10 == 0:
                    loss = agent.train(batch_size)
                
                obs = next_obs
                step += 1
                if step % 100 == 0:
                    print(f"  Step: {step}")
            
            # Update target network periodically
            if (episode + 1) % update_target_freq == 0:
                agent.update_target_network()
            
            # Decay exploration rate
            agent.decay_epsilon()
            
            episode_rewards.append(episode_reward)
        
            # Save model every episode
            agent.q_network.model.save(AGENTS_SAVE_PATH + f"_agent_{agent_idx}.h5")
        

            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1:3d}/{episodes} | "
                    f"Reward: {episode_reward:8.2f} | "
                    f"Avg (10): {avg_reward:8.2f} | "
                    f"ε: {agent.epsilon:.3f} | "
                    f"Buffer: {len(agent.replay_buffer)} | "
                    f"Agent: {agent_idx + 1}/{len(aggregator_instance.agents)}")
            print(f"  ✓ Model saved to {AGENTS_SAVE_PATH}_agent_{agent_idx}.h5")

            print("\nGenerating training curve...")
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(episode_rewards, label='Episode Reward', alpha=0.7)
            plt.plot(np.convolve(episode_rewards, np.ones(10)/10, mode='valid'), 
                    label='Moving Avg (10)', linewidth=2, color='orange')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Training Rewards')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.plot(agent.loss_history, label='Training Loss', alpha=0.7)
            plt.xlabel('Training Step')
            plt.ylabel('MSE Loss')
            plt.title('Training Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'training_results_{agent_idx}.png', dpi=100, bbox_inches='tight')
            print("✓ Training curve saved to training_results.png")
            
            episode_rewards_list[agent_idx] = episode_rewards
        if (episode + 1) % EPISODES_PER_AGGREGATION == 0:
            print("\nAggregating agent models using FedAVG...")
            aggregator_instance.aggregate()
            print("✓ Aggregation completed.")
            aggregator_instance.save_model(MODEL_SAVE_PATH)
            print(f"✓ Aggregated model saved to {MODEL_SAVE_PATH}")
    print("-" * 60)
    print("\n✓ Training completed!")
    
    
    
    print("\n" + "="*60)
    print(f"Final Results:")
    print(f"  - Final Episode Reward: {episode_rewards[-1]:.2f}")
    print(f"  - Average Last 10 Episodes: {np.mean(episode_rewards[-10:]):.2f}")
    print(f"  - Best Episode: {np.max(episode_rewards):.2f}")
    print(f"  - Model saved at: {MODEL_SAVE_PATH}")
    print("="*60)
