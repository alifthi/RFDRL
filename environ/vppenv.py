import gym
from CONFIG import LOW, HIGH, OBSERVATION_SPACE_DIM, NUM_EVS, ACTION_PER_PILE, PER_EV_POWER
import numpy as np
class VPPEnv(gym.Env):
    """
    A custom OpenAI Gym environment for the VPP (Virtual Power Plant) simulation.
    """

    def __init__(self, config, data):
        super(VPPEnv, self).__init__()
        self.config = config
        self.data = data
        self.evs = []
        self.modes = []
        self.departed_evs = []
        self.dt = 5/60
        self.ev_cap = 100.0
        self._setup_state()

    def step(self, action):
        """
        Execute one time step within the environment.
        """
        indices = np.nonzero(action)[0]
        self.modes = [self.decode_action(idx) for idx in indices]
        for i, ev in enumerate(self.evs):
            ev.soc += self.modes[i] * PER_EV_POWER*self.dt / self.ev_cap
            ev.soc = np.clip(ev.soc, 0.0, 1.0)
            ev.stay -= 1
            if ev.stay <= 0:
                self.departed_evs.append(i)
        if self.departed_evs:
            self._req_for_evs()
                

    def reset(self):
        """
        Reset the state of the environment to an initial state.
        """
        pass

    def render(self, mode='human'):
        """
        Render the environment to the screen.
        """
        pass

    def close(self):
        """
        Perform any necessary cleanup.
        """
        pass
    
    def _setup_state(self):
        """
        Initialize the environment state.
        """
        self.observation_space = gym.spaces.Box(low=LOW, high=HIGH, shape=(OBSERVATION_SPACE_DIM,), dtype=float)
        
    def _setup_action(self):
        """
        Initialize the action space.
        """
        self.action_space = gym.spaces.Discrete(3*NUM_EVS)  
        
    def _update_evs(self, evs):
        """
        Update electric vehicles.
        """
        self.evs = evs
    
    def decode_action(self, one_idx):
        mode = one_idx % ACTION_PER_PILE - 1
        return mode
    
    def _req_for_evs(self):
        """
        Request new electric vehicles to replace departed ones.
        """
        for idx in self.departed_evs:
            # self.evs[idx] = 
            pass
    
    def _get_obs(self):
        """
        Create the observation for the current state.
        """
        pass
    
    def _man_observation(self):
        """
        Manipulate the observation vector.
        """
        pass
    
    def _get_reward(self):
        """
        Calculate the reward for the current step.
        """
        pass
        