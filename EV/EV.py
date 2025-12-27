import numpy as np
from CONFIG import STEP_TIME
class EV:
    def __init__(self):
        self.soc = None  
        self.stay = None
        self.step_time = STEP_TIME
        self._init_ev()
        
    def _init_ev(self):
        self.soc = np.random.normal(0.5, 0.1)
        self.stay  = np.random.normal(24, 1) // self.step_time