import gym
from CONFIG import LOW, HIGH, OBSERVATION_SPACE_DIM, NUM_EVS, ACTION_PER_PILE, PER_EV_POWER, STEP_TIME
import numpy as np
from EV.EVQueue import EVQueue
class VPPEnv(gym.Env):
    """
    A custom OpenAI Gym environment for the VPP (Virtual Power Plant) simulation.
    """

    def __init__(self, data):
        super(VPPEnv, self).__init__()
        self.data = data
        self.modes = []
        self.departed_evs = []
        self.t = 0
        self.dt = STEP_TIME
        self.ev_cap = 100.0
        self._setup_state()
        self.evqueue = EVQueue()
        self.evs = [self.evqueue._get_ev() for _ in range(NUM_EVS)]

    def step(self, action):
        """
        Execute one time step within the environment.
        """
        done = False
        indices = np.nonzero(action)[0]
        self.modes = [self.decode_action(idx) for idx in indices]
        for i, ev in enumerate(self.evs):
            ev.soc += self.modes[i] * PER_EV_POWER*self.dt / self.ev_cap
            ev.soc = np.clip(ev.soc, 0.0, 1.0)
            ev.stay -= 1
            if ev.stay <= 0:
                self.departed_evs.append(i)
        reward = self._get_reward()
        if self.departed_evs:
            self._req_for_evs()
        obs = self._get_obs()
        self.t += 1
        if self.t == 35040:
            done = True
        return reward, obs,  done
                
    def reset(self):
        """
        Reset the state of the environment to an initial state.
        """
        self.t = 0

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
            self.evs[idx] = self.evqueue._get_ev()
        self.departed_evs = []
    
    def _get_obs(self):
        """
        Build the observation vector and store it in self.obs.
        Observation layout (default):
        [ pv_norm, wind_norm, load_norm, price_norm,
            n_ev_frac, mean_soc, mean_stay_frac,
            pile0_frac, pile1_frac, pile2_frac, pile3_frac,
            sin_hour, cos_hour, sin_day, cos_day, ...pad/trunc... ]

        - pile definition (paper-aligned default):
            pile0 = low SOC, urgent
            pile1 = low SOC, non-urgent
            pile2 = high SOC, urgent
            pile3 = high SOC, non-urgent
        where "low" = SOC < 0.5 and "urgent" = remaining stay <= urgent_threshold_hours.
        - Normalizations chosen robustly from dataset ranges (fallbacks included).
        """

        # ----- defensive defaults -----
        # ensure time index exists
        self.t = getattr(self, "t", 0)

        # dataset arrays (defensive)
        p_pv = float(self.data.pv[self.t]) 
        p_wind = float(self.data.wind[self.t])
        p_load = float(self.data.load[self.t])
        price  = float(self.data.price[self.t])

        # ----- normalization constants (automatic, robust) -----
        # P_nom: nominal power scale (kW). use dataset max or fallback 100 kW
        try:
            P_nom = float(max(
                1.0,
                np.nanmax(self.data.load),
                np.nanmax(self.data.pv),
                np.nanmax(self.data.wind)
            ))
        except Exception:
            P_nom = 100.0

        # Price nominal
        try:
            PRICE_NOM = float(max(1e-3, np.nanmax(self.data.price)))
        except Exception:
            PRICE_NOM = max(1.0, price)

        # ----- EV pool summaries -----
        n_ev = len(self.evs) if self.evs is not None else 0

        if n_ev > 0:
            socs = np.array([ev.soc for ev in self.evs], dtype=float)
            stays = np.array([ev.stay for ev in self.evs], dtype=float)  # steps remaining
            mean_soc = float(np.nanmean(socs))
            # convert remaining stay (steps) -> hours
            step_hours = float(getattr(self, "dt", self.dt)) if hasattr(self, "dt") else float(self.dt)
            stays_hours = stays * step_hours
            mean_stay_hours = float(np.nanmean(stays_hours))
            # robust stats
            median_stay_hours = float(np.nanmedian(stays_hours))
        else:
            mean_soc = 0.0
            mean_stay_hours = 0.0
            median_stay_hours = 0.0

        # ----- pile construction (paper-aligned default) -----
        # thresholds (tunable)
        SOC_THRESH = 0.5              # low vs high SOC threshold (fraction)
        URGENT_HOURS = 6.0            # urgency threshold (hours). EVs with <= this are 'urgent'
        pile_counts = [0, 0, 0, 0]    # low-urgent, low-non, high-urgent, high-non

        if n_ev > 0:
            for ev in self.evs:
                soc = float(ev.soc)
                stay_hours = float(ev.stay) * step_hours
                low = soc < SOC_THRESH
                urgent = stay_hours <= URGENT_HOURS
                if low and urgent:
                    pile_counts[0] += 1
                elif low and not urgent:
                    pile_counts[1] += 1
                elif (not low) and urgent:
                    pile_counts[2] += 1
                else:
                    pile_counts[3] += 1

        # fractions (avoid div by zero)
        if n_ev > 0:
            pile_fracs = [c / n_ev for c in pile_counts]
        else:
            pile_fracs = [0.0, 0.0, 0.0, 0.0]

        # ----- time features (smooth) -----
        # steps per day = 24h / dt_hours
        dt_hours = float(getattr(self, "dt", self.dt))
        steps_per_day = max(1, int(round(24.0 / dt_hours)))
        steps_per_week = steps_per_day * 7

        step_of_day = int(self.t % steps_per_day)
        step_of_week = int(self.t % steps_per_week)

        hour_angle = 2.0 * np.pi * (step_of_day / float(steps_per_day))
        day_angle  = 2.0 * np.pi * (step_of_week / float(steps_per_week))

        sin_hour = np.sin(hour_angle)
        cos_hour = np.cos(hour_angle)
        sin_day  = np.sin(day_angle)
        cos_day  = np.cos(day_angle)

        # ----- build vector and normalize -----
        obs_vec = [
            p_pv / P_nom,               # normalized pv
            p_wind / P_nom,             # normalized wind
            p_load / P_nom,             # normalized load
            price / PRICE_NOM,          # normalized price
            (n_ev / float(max(1, NUM_EVS))),  # fraction of modeled EV slots occupied
            mean_soc,                   # SOC in [0,1]
            median_stay_hours / 24.0,   # normalize by 24h -> ~fraction of a day
            pile_fracs[0],
            pile_fracs[1],
            pile_fracs[2],
            pile_fracs[3],
            sin_hour,
            cos_hour,
            sin_day,
            cos_day
        ]

        # ----- pad or truncate to OBSERVATION_SPACE_DIM -----
        out_dim = OBSERVATION_SPACE_DIM
        obs = np.asarray(obs_vec, dtype=np.float32)
        if obs.shape[0] < out_dim:
            # pad with zeros
            pad = np.zeros(out_dim - obs.shape[0], dtype=np.float32)
            obs = np.concatenate([obs, pad])
        elif obs.shape[0] > out_dim:
            obs = obs[:out_dim]

        return obs

    def _man_observation(self, obs):
        """
        Manipulate the observation vector.
        """
        return obs
    
    def _get_reward(self):
        reward = 0.0

        ev_power = 0.0
        for mode in self.modes:
            ev_power += mode * PER_EV_POWER

        p_net = (
            self.data.load[self.t]
            - self.data.pv[self.t]
            - self.data.wind[self.t]
            + ev_power
        )

        price = self.data.price[self.t]
        reward -= price * p_net * self.dt

        SOC_TARGET = 1.0
        beta = 10.0

        for idx in self.departed_evs:
            ev = self.evs[idx]
            if ev.soc < SOC_TARGET:
                reward -= beta * (SOC_TARGET - ev.soc) ** 2

        return reward

        