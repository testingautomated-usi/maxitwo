import gym
import numpy as np
from gym import spaces

from envs.udacity.config import MAX_STEERING, INPUT_DIM
from global_log import GlobalLog
from test_generators.mapelites.individual import Individual
from custom_types import ObserveData


class MockGymEnv(gym.Env):
    """
    Gym interface for DonkeyCar with support for using
    a VAE encoded observation instead of raw pixels if needed.
    """

    def __init__(self):

        self.logger = GlobalLog('MockGymEnv')

        # steering + throttle, action space must be symmetric
        self.action_space = spaces.Box(
            low=np.array([-MAX_STEERING, -1]),
            high=np.array([MAX_STEERING, 1]),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(low=0, high=255, shape=INPUT_DIM, dtype=np.uint8)

    def step(self, action: np.ndarray) -> ObserveData:
        """
        :param action: (np.ndarray)
        :return: (np.ndarray, float, bool, dict)
        """
        # action[0] is the steering angle
        # action[1] is the throttle

        return self.observation_space.sample(), np.random.choice(a=[True, False]), {}

    def reset(self, skip_generation: bool = False, individual: Individual = None) -> np.ndarray:
        return self.observation_space.sample()

    def render(self, mode='human'):
        """
        :param mode: (str)
        """
        if mode == 'rgb_array':
            return self.observation_space.sample()
        return None

    def close(self):
        pass
