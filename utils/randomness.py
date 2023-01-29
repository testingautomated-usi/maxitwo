import random

import numpy as np
import tensorflow as tf

from global_log import GlobalLog


def set_random_seed(seed: int) -> None:
    log = GlobalLog("set_random_seed")
    log.info("Setting random seed: {}".format(seed))
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # Seed tensorflow RNG
    tf.random.set_seed(seed)
