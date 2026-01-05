""" Data path """
HH_LOAD = "RFDRL/data/households_load_profile.csv"
PRICE = "RFDRL/data/market_prices_2020_profile.csv"
PV_POWER = "RFDRL/data/PV_load_2020_profile.csv"
WT_POWER = "RFDRL/data/WT_load_2020_profile.csv"


""" ENV setup """
# Obsetvation space variables
OBSERVATION_SPACE_DIM = 11
LOW = -1
HIGH = 1

MODEL_SAVE_PATH = "trained_models/dqn_agent_checkpoint.h5"

NUM_EVS = 4

ACTION_PER_PILE = 3

PER_EV_POWER = 3.7

STEP_TIME = 15/60


LAMBDA1 = 1e-4
LAMBDA2 = 1e-4

NUM_AGENTS = 5
EPISODES_PER_AGGREGATION = 2

AGENTS_SAVE_PATH = "trained_models/aggregated_dqn_agent"