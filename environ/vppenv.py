import gym
from CONFIG import LOW, HIGH, OBSERVATION_SPACE_DIM, NUM_EVS, ACTION_PER_PILE, PER_EV_POWER, STEP_TIME
import numpy as np
from EV.EVQueue import EVQueue
import gym
from CONFIG import LOW, HIGH, OBSERVATION_SPACE_DIM, NUM_EVS, ACTION_PER_PILE, PER_EV_POWER, STEP_TIME
import numpy as np
from EV.EVQueue import EVQueue


class VPPEnv(gym.Env):
    """
    Improved VPP environment with:
      - gym API fixes (step -> obs, reward, done, info), reset returns obs
      - reset initializes cumulative counters (used by terminal reward)
      - flexible action parsing (accepts per-ev actions or discrete/one-hot)
      - info dict returned from step for debugging / training
      - seed() support
      - minor defensive checks and documentation
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, data):
        super(VPPEnv, self).__init__()
        self.data = data
        self.modes = [0] * NUM_EVS
        self.departed_evs = []
        self.t = 0
        self.dt = STEP_TIME
        self.ev_cap = 100.0

        # cum counters used by terminal reward (kept across steps in an episode)
        self.cum_grid_energy = 0.0
        self.cum_excess_renew = 0.0
        self.cum_grid_cost = 0.0

        # EV pool
        self.evqueue = EVQueue()
        self.evs = [self.evqueue._get_ev() for _ in range(NUM_EVS)]

        # gym spaces
        self._setup_state()
        self._setup_action()

        # bookkeeping
        self.done = False

    # ---------------- gym API ----------------
    def step(self, action):
        """
        Accepts several action formats for convenience:
          - array-like of length NUM_EVS with values in {-1,0,1} (per-ev mode)
          - int or array representing one-hot indices (legacy from previous implementation)

        Returns: (obs, reward, done, info)
        """
        # --- parse action into per-ev modes list ---
        modes = [0] * NUM_EVS
        # if action is an array-like per-ev (length NUM_EVS)
        try:
            a = np.asarray(action)
            if a.shape == ():
                # scalar (could be discrete index / legacy)
                idxs = np.nonzero(action)[0] if np.ndim(action) > 0 else None
            elif a.size == NUM_EVS:
                # per-ev modes provided
                modes = [int(x) for x in a.tolist()]
            else:
                # fallback: treat as indices where action != 0
                idxs = np.nonzero(a)[0]
        except Exception:
            idxs = None

        # if we didn't fill modes yet but have nonzero indices, decode them
        if sum(abs(int(m)) for m in modes) == 0:
            # legacy: action given as indices or one-hot
            try:
                if isinstance(action, (list, tuple, np.ndarray)):
                    arr = np.asarray(action)
                    idxs = np.nonzero(arr)[0]
                elif isinstance(action, int):
                    # single integer -- interpret as one-hot index
                    idxs = [int(action)]
            except Exception:
                idxs = None

            if idxs is not None:
                # map each index to an EV slot and decode mode
                for idx in idxs:
                    ev_slot = int(idx) // ACTION_PER_PILE  # backward-compatible mapping
                    mode = self.decode_action(idx)
                    if 0 <= ev_slot < NUM_EVS:
                        modes[ev_slot] = mode

        # finalize modes and store
        self.modes = modes

        # --- apply modes to EVs ---
        for i, ev in enumerate(self.evs):
            mode = int(self.modes[i])
            ev.soc += mode * PER_EV_POWER * self.dt / self.ev_cap
            ev.soc = np.clip(ev.soc, 0.0, 1.0)
            ev.stay -= 1
            if ev.stay <= 0:
                self.departed_evs.append(i)

        # --- compute reward at current timestep ---
        reward = self._get_reward()

        # if EVs departed, request replacements
        if self.departed_evs:
            self._req_for_evs()

        # --- observation for next state ---
        obs = self._get_obs()

        # increment time and check terminal condition
        self.t += 1
        if self.t >= 35040:
            self.done = True
        else:
            self.done = False

        # info dict is helpful for monitoring/training
        info = {
            "t": self.t,
            "ev_power": sum([m * PER_EV_POWER for m in self.modes]),
            "departed_count": len(self.departed_evs),
            "cum_grid_energy": self.cum_grid_energy,
            "cum_excess_renew": self.cum_excess_renew,
            "cum_grid_cost": self.cum_grid_cost,
        }

        return obs, reward, self.done, info

    def reset(self, *, seed=None, options=None):
        """
        Reset environment state and return initial observation.
        """
        if seed is not None:
            self.seed(seed)

        self.t = 0
        self.done = False
        self.departed_evs = []

        # reset cumulative episode counters
        self.cum_grid_energy = 0.0
        self.cum_excess_renew = 0.0
        self.cum_grid_cost = 0.0

        # reset EV queue and EVs
        self.evqueue = EVQueue()
        self.evs = [self.evqueue._get_ev() for _ in range(NUM_EVS)]
        self.modes = [0] * NUM_EVS

        obs = self._get_obs()
        return obs

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        np.random.seed(seed)

    # ---------------- helpers / setup ----------------
    def _setup_state(self):
        """
        Initialize observation space.
        """
        self.observation_space = gym.spaces.Box(low=LOW, high=HIGH, shape=(OBSERVATION_SPACE_DIM,), dtype=float)

    def _setup_action(self):
        """
        Initialize action space. Keep compatibility with legacy discrete scheme,
        but documents that agents are encouraged to pass per-ev arrays of length NUM_EVS.
        """
        # legacy discrete / one-hot representation
        self.action_space = gym.spaces.Discrete(ACTION_PER_PILE * NUM_EVS)

    def _update_evs(self, evs):
        self.evs = evs

    def decode_action(self, one_idx):
        mode = one_idx % ACTION_PER_PILE - 1
        return mode

    def _req_for_evs(self):
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
        """
        Reward implementing the piecewise step and terminal rewards from
        Feng et al., Applied Energy (2023), Eqs. (18)-(23).
        See: reward composition (step + trajectory-ending). :contentReference[oaicite:4]{index=4}
        """
        reward = 0.0

        # --- compute EV power (same as your original) ---
        ev_power = 0.0
        for mode in self.modes:
            ev_power += mode * PER_EV_POWER

        # net power: positive -> imported from grid; negative -> exported to grid
        p_net = (
            self.data.load[self.t]
            - self.data.pv[self.t]
            - self.data.wind[self.t]
            + ev_power
        )

        price = self.data.price[self.t]
        dt = getattr(self, "dt", 1.0)  # hours (original code used self.dt)

        # --- Immediate cost (keep original immediate grid cost) ---
        # price * p_net * dt: if p_net>0 cost, if p_net<0 income (negative cost)
        reward -= price * p_net * dt
        # also track per-step grid energy / export so we can build trajectory totals
        eg_step = max(0.0, p_net) * dt         # energy acquired from grid this step (kWh)
        eer_step = max(0.0, -p_net) * dt      # excess renewable exported (kWh)
        cg_step = price * eg_step              # cost of grid electricity this step (monetary)

        # initialize cumulative episode counters if missing
        if not hasattr(self, "cum_grid_energy"):
            self.cum_grid_energy = 0.0
        if not hasattr(self, "cum_excess_renew"):
            self.cum_excess_renew = 0.0
        if not hasattr(self, "cum_grid_cost"):
            self.cum_grid_cost = 0.0

        self.cum_grid_energy += eg_step
        self.cum_excess_renew += eer_step
        self.cum_grid_cost += cg_step

        # --- Step reward: total load value L_t piecewise (Eq. 18) ---
        # Map L_t -> p_net (kW). Reward peaks at 0 and declines away from 0.
        L = p_net  # in kW
        if L < -1.0:
            L_reward = L + 1.0
        elif -1.0 <= L < 0.0:
            L_reward = 15.0 * L + 15.0
        elif 0.0 <= L < 1.0:
            L_reward = -15.0 * L + 15.0
        else:  # L >= 1.0
            L_reward = -L + 1.0

        reward += L_reward

        # --- Step reward: available energy for each EV WHEN DEPARTING (Eq. 19) ---
        # You used self.departed_evs earlier; apply paper piecewise reward per departed EV.
        for idx in self.departed_evs:
            ev = self.evs[idx]
            soc_pct = float(ev.soc) * 100.0  # convert [0..1] -> percent
            if 0.0 <= soc_pct < 90.0:
                depart_reward = 5.0 * soc_pct - 300.0
            elif 90.0 <= soc_pct < 100.0:
                depart_reward = -10.0 * soc_pct + 1050.0
            else:
                # corner-case if soc_pct >=100 or negative; clamp to last segment
                depart_reward = -10.0 * min(soc_pct, 100.0) + 1050.0
            reward += depart_reward

        # --- Trajectory-ending (terminal) rewards (Eqs. 20-23) ---
        # Apply only when episode ends. We attempt to detect 'done' robustly.
        episode_done = getattr(self, "done", False)
        try:
            # also check typical terminal-index condition if available
            if not episode_done and hasattr(self, "data") and hasattr(self.data, "price"):
                if self.t >= (len(self.data.price) - 1):
                    episode_done = True
        except Exception:
            pass

        if episode_done:
            # (a) average SOC at departure -> use average SOC across EVs (percent)
            try:
                ev_socs = [float(ev.soc) * 100.0 for ev in self.evs if hasattr(ev, "soc")]
                avg_soc_pct = float(sum(ev_socs)) / max(1.0, len(ev_socs))
            except Exception:
                avg_soc_pct = 0.0

            # Eq (20): average available energy when EV departs (piecewise)
            if 0.0 <= avg_soc_pct < 75.0:
                term_avg_soc = (4.0 / 25.0) * avg_soc_pct - 9.0
            elif 75.0 <= avg_soc_pct < 100.0:
                term_avg_soc = (-2.0 / 25.0) * avg_soc_pct + 9.0
            else:
                term_avg_soc = (-2.0 / 25.0) * min(avg_soc_pct, 100.0) + 9.0
            reward += term_avg_soc

            # (b) energy acquired from the grid Eg (kWh) -> use cumulative
            Eg = getattr(self, "cum_grid_energy", 0.0)
            # Eq (21)
            if 0.0 <= Eg < 800.0:
                term_Eg = -25.0 * Eg + 20000.0
            else:
                term_Eg = -Eg + 800.0
            reward += term_Eg

            # (c) excess renewable energy output Eer (kWh) -> cumulative
            Eer = getattr(self, "cum_excess_renew", 0.0)
            # Eq (22)
            if 0.0 <= Eer < 3000.0:
                term_Eer = -(5.0 / 3.0) * Eer + 5000.0
            else:
                term_Eer = -(3.0 / 2.0) * Eer + 4500.0
            reward += term_Eer

            # (d) cost of grid electricity Cg -> cumulative
            Cg = getattr(self, "cum_grid_cost", 0.0)
            # Eq (23)
            if 0.0 <= Cg < 450.0:
                term_Cg = -40.0 * Cg + 18000.0
            else:
                term_Cg = -10.0 * Cg + 4500.0
            reward += term_Cg

            # (optional) clear cumulative counters so next episode starts fresh
            # (only if your env expects them reset here)
            try:
                self.cum_grid_energy = 0.0
                self.cum_excess_renew = 0.0
                self.cum_grid_cost = 0.0
            except Exception:
                pass

        return reward

            