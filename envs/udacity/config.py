####################
# Udacity params #
####################

MAX_SPEED_UDACITY = 34  # needed to make the car goes to 30 km/h max
MIN_SPEED_UDACITY = 15

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 160
N_CHANNELS = 3
INPUT_DIM = (IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNELS)

# Symmetric command
MAX_STEERING = 1
MIN_STEERING = -MAX_STEERING
MAX_STEERING_DIFF = 0.2

# Max cross track error (used in normal mode to reset the car)
MAX_CTE_ERROR = 3.0

BASE_PORT = 4567

# PID constants
KP_UDACITY = 1.0
KD_UDACITY = 0.2
KI_UDACITY = 0.0
