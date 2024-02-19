# Input dimension for VAE
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 160
N_CHANNELS = 3
INPUT_DIM = (IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNELS)

# Symmetric command
MAX_STEERING = 1
MIN_STEERING = -MAX_STEERING

# Simulation config
MIN_THROTTLE = 0.0
# max_throttle: 0.6 for level 0 and 0.5 for level 1
MAX_THROTTLE = 1.0
# Max cross track error (used in normal mode to reset the car)
MAX_CTE_ERROR = 3.0

THROTTLE_REWARD_WEIGHT = 0.1

BASE_PORT = 64256
MAP_SIZE = 250
BEAMNG_VERSION = "0.23"

MAX_SPEED_BEAMNG = 30
MIN_SPEED_BEAMNG = 15
