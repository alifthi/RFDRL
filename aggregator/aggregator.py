from agents.dqn_agent import DQNAgent
from CONFIG import NUM_EVS, ACTION_PER_PILE, NUM_AGENTS
import numpy as np
class aggregator:
    def __init__(self):
        self.agents = []
        
        self.main_model = DQNAgent(
                            obs_dim=11,
                            num_evs=NUM_EVS,
                            num_actions=ACTION_PER_PILE,
                            hidden_dims=[64, 128, 64],
                            learning_rate=1e-3,
                            gamma=0.99)
        
        self.init_agents()
        
    def aggregate(self):
        '''FedAVG aggregation of agents' model weights'''
        main_weights = self.main_model.q_network.model.get_weights()
        new_weights = [np.zeros_like(w, dtype=np.float64) for w in main_weights]
        
        for agent in self.agents:
            agent_weights = agent.q_network.model.get_weights()
            for i in range(len(new_weights)):
                new_weights[i] += agent_weights[i] / NUM_AGENTS
        
        self.main_model.q_network.model.set_weights(new_weights)
        
        self.set_agents_weights(new_weights)
        

    def init_agents(self):
        for _ in range(NUM_AGENTS):
            agent = DQNAgent(
                    obs_dim=11,
                    num_evs=NUM_EVS,
                    num_actions=ACTION_PER_PILE,
                    hidden_dims=[64, 128, 64],
                    learning_rate=1e-3,
                    gamma=0.99)
            self.agents.append(agent)
        self.aggregate()
    def set_agents_weights(self, weights):
        for agent in self.agents:
            agent.q_network.model.set_weights(weights)
    def save_model(self, path):
        self.main_model.q_network.model.save(path)