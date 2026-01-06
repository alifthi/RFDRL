import os

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use("Agg")

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from utils.load_data import load_data
from environ.vppenv import VPPEnv
from aggregator.aggregator import aggregator
from CONFIG import EPISODES_PER_AGGREGATION, NUM_EVS, ACTION_PER_PILE, MODEL_SAVE_PATH, AGENTS_SAVE_PATH

# ---- Helper function: single-agent episode runner (runs in a thread) ----
def run_agent_episode(agent, env, episode_idx, max_steps, batch_size, update_target_freq):
    """
    Run a single episode for `agent` in `env`.
    Returns a dictionary with stats and small artifacts (episode_reward, agent_idx is in agent object).
    """
    try:
        obs = env.reset()
        if obs is None:
            # fallback observation shape — keep the same as your previous code
            obs = np.zeros(11, dtype=np.float32)

        done = False
        step = 0
        episode_reward = 0.0

        # optional: step-wise loss tracking done inside agent.train() and agent.loss_history
        while not done and step < max_steps:
            actions = agent.select_action(obs, training=True)  # expected to return iterable of action indices per EV

            # convert actions to one-hot for environment as original code
            action_onehot = np.zeros(NUM_EVS * ACTION_PER_PILE, dtype=np.float32)
            for ev_idx, action_idx in enumerate(actions):
                action_onehot[ev_idx * ACTION_PER_PILE + action_idx] = 1

            next_obs, reward, done, _ = env.step(action_onehot)
            if next_obs is None:
                next_obs = np.zeros_like(obs)

            episode_reward += reward

            # store transition — keep the same API
            agent.remember(obs, actions, reward, next_obs, done)

            # train periodically (your code trained each 10 steps)
            if step % 10 == 0:
                # agent.train should internally return loss (or None)
                _ = agent.train(batch_size)

            obs = next_obs
            step += 1
            if step % 100 == 0:
                print(f"    Agent {agent.agent_index}  Step: {step}")

        # update target network according to your frequency (do it per-episode here)
        if (episode_idx + 1) % update_target_freq == 0:
            agent.update_target_network()

        # decay epsilon once per completed episode
        agent.decay_epsilon()

        # Save agent model
        if not os.path.exists(os.path.dirname(AGENTS_SAVE_PATH)) and os.path.dirname(AGENTS_SAVE_PATH):
            os.makedirs(os.path.dirname(AGENTS_SAVE_PATH), exist_ok=True)

        # This will be the per-agent file name (agent should have index attribute or infer elsewhere)
        # We'll assume the agent object carries a .agent_index attribute set before starting threads.
        agent_model_path = AGENTS_SAVE_PATH + f"_agent_{agent.agent_index}.h5"
        agent.q_network.model.save(agent_model_path)

        stats = {
            "agent_index": agent.agent_index,
            "episode_reward": episode_reward,
            "steps": step,
            "model_path": agent_model_path,
            "loss_history": list(agent.loss_history) if hasattr(agent, "loss_history") else [],
            "replay_buffer_len": len(agent.replay_buffer) if hasattr(agent, "replay_buffer") else None,
            "epsilon": getattr(agent, "epsilon", None),
        }
        return stats

    except Exception as e:
        # Return exception info to the caller to avoid silent thread death
        return {"error": str(e), "agent_index": getattr(agent, "agent_index", None)}

# ---- Plot helper ----
def save_training_plots(agent_idx, episode_rewards, loss_history):
    plt.figure(figsize=(12, 5))

    # Reward plot
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, label='Episode Reward', alpha=0.7)
    if len(episode_rewards) >= 10:
        ma = np.convolve(episode_rewards, np.ones(10) / 10, mode='valid')
        plt.plot(np.arange(9, 9 + len(ma)), ma, label='Moving Avg (10)', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Agent {agent_idx} Training Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(loss_history, label='Training Loss', alpha=0.7)
    plt.xlabel('Training Step')
    plt.ylabel('MSE Loss')
    plt.title(f'Agent {agent_idx} Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'training_results_agent_{agent_idx}.png'
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    return filename

# ---- Main parallel training routine ----
if __name__ == "__main__":
    print("=" * 60)
    print("Parallel DQN Agent Training for VPP Environment (Thread-based)")
    print("=" * 60)

    # 1) Load data
    print("\n[1/5] Loading data...")
    data = load_data()
    print("✓ Data loaded successfully")

    # 2) Create aggregator and initialize agents
    print("\n[2/5] Initializing aggregator and agents...")
    aggregator_instance = aggregator()
    agents = aggregator_instance.agents  # list of agent objects
    num_agents = len(agents)
    print(f"✓ Found {num_agents} agents in aggregator")

    # Attach agent_index to agents so worker function can save models with unique names
    for idx, a in enumerate(agents):
        setattr(a, "agent_index", idx)

    # 3) Create independent environments — one per agent
    print("\n[3/5] Creating per-agent environments...")
    envs = [VPPEnv(data) for _ in range(num_agents)]
    print(f"✓ Created {len(envs)} environment instances")

    # 4) Load global model weights into agents if aggregated model exists
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"\n✓ Loading pre-trained global model from {MODEL_SAVE_PATH}...")
        # load into aggregator main model, then push to agents
        aggregator_instance.main_model.q_network.model.load_weights(MODEL_SAVE_PATH)
        global_weights = aggregator_instance.main_model.q_network.model.get_weights()
        aggregator_instance.set_agents_weights(global_weights)
        print("✓ Global model loaded and distributed to agents")
    else:
        print(f"\n! No pre-trained global model found at {MODEL_SAVE_PATH} — starting from scratch")

    # 5) Training loop (parallel per-episode across agents)
    print("\n[4/5] Starting parallel training...")
    episodes = 200
    batch_size = 128
    update_target_freq = 10
    max_steps = 35050

    # keep per-agent reward histories and loss histories
    episode_rewards_list = [[] for _ in range(num_agents)]
    loss_histories = [[] for _ in range(num_agents)]

    # ThreadPool with number of workers equal to number of agents
    with ThreadPoolExecutor(max_workers=num_agents) as executor:
        for episode in range(episodes):
            # submit one episode job per agent (they run concurrently)
            futures = []
            for agent_idx, agent in enumerate(agents):
                env = envs[agent_idx]
                fut = executor.submit(
                    run_agent_episode,
                    agent,
                    env,
                    episode,
                    max_steps,
                    batch_size,
                    update_target_freq,
                )
                futures.append(fut)

            # collect results
            for fut in as_completed(futures):
                res = fut.result()
                if "error" in res:
                    print(f"[Error] Agent {res.get('agent_index')} raised: {res['error']}")
                    continue

                idx = res["agent_index"]
                episode_reward = res["episode_reward"]
                episode_rewards_list[idx].append(episode_reward)

                # extend loss history if returned
                if res.get("loss_history") is not None:
                    # append last-known loss history snapshot; agent keeps full history in memory
                    loss_histories[idx] = res["loss_history"]

                # print a concise per-agent per-episode line
                avg10 = np.mean(episode_rewards_list[idx][-10:]) if len(episode_rewards_list[idx]) >= 1 else 0.0
                eps_val = res.get("epsilon", None)
                buffer_len = res.get("replay_buffer_len", None)
                print(
                    f"[Episode {episode + 1}/{episodes}] Agent {idx + 1}/{num_agents} | "
                    f"Reward: {episode_reward:8.2f} | Avg(10): {avg10:8.2f} | "
                    f"ε: {eps_val:.3f} | Buffer: {buffer_len}"
                )

                # Save per-agent training plot every N episodes to reduce IO (here: every aggregation)
                # (plots are cheap; change freq if you wish)
                # if (episode + 1) % EPISODES_PER_AGGREGATION == 0 or (episode + 1) == episodes:
                plot_file = save_training_plots(idx, episode_rewards_list[idx], loss_histories[idx])
                print(f"  ✓ Agent {idx} training plot saved to {plot_file}")

            # After all agents completed this episode, optionally aggregate every EPISODES_PER_AGGREGATION
            if (episode + 1) % EPISODES_PER_AGGREGATION == 0:
                print(f"\n[{datetime.now().isoformat()}] Aggregating agent models using FedAVG (round {episode + 1}) ...")
                try:
                    aggregator_instance.aggregate()
                    aggregator_instance.save_model(MODEL_SAVE_PATH)
                    print(f"✓ Aggregated model saved to {MODEL_SAVE_PATH}")
                    # After aggregation, distribute global weights to agents so next rounds start from global
                    global_weights = aggregator_instance.main_model.q_network.model.get_weights()
                    aggregator_instance.set_agents_weights(global_weights)
                except Exception as e:
                    print(f"[Aggregation error] {e}")

    print("-" * 60)
    print("\n✓ Parallel training completed!")

    # Final summary for each agent
    for idx in range(num_agents):
        eps_rewards = episode_rewards_list[idx]
        if not eps_rewards:
            continue
        print(
            f"Agent {idx}: Final Reward {eps_rewards[-1]:.2f} | "
            f"Avg Last 10: {np.mean(eps_rewards[-10:]):.2f} | "
            f"Best: {np.max(eps_rewards):.2f} | "
            f"Saved Model: {AGENTS_SAVE_PATH}_agent_{idx}.h5"
        )

    # Final aggregated model location note
    print(f"\nAggregated model (if created) saved at: {MODEL_SAVE_PATH}")
    print("=" * 60)
