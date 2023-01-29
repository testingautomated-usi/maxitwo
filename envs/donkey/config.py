####################
# DonkeyCar params #
####################

MAX_SPEED_DONKEY = 38  # needed to make the car goes to 30 km/h max
# MAX_SPEED_DONKEY = 45
MIN_SPEED_DONKEY = 15

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 160
N_CHANNELS = 3
INPUT_DIM = (IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNELS)

# Symmetric command
MAX_STEERING = 1
MIN_STEERING = -MAX_STEERING
MAX_STEERING_DIFF = 0.2

# Max cross track error (used in normal mode to reset the car)
MAX_CTE_ERROR = 2.5

BASE_PORT = 9091
BASE_SOCKET_LOCAL_ADDRESS = 52804

# PID constants
KP_DONKEY = 0.7
KD_DONKEY = 0.0
KI_DONKEY = 0.0
