from typing import Dict

import numpy as np

from custom_types import GymEnv
from self_driving.agent import Agent


class RandomAgent(Agent):
    def __init__(self, env: GymEnv, env_name: str):
        super().__init__(env=env, env_name=env_name)

    def predict(self, obs: np.ndarray, state: Dict) -> np.ndarray:
        return self.env.action_space.sample()
