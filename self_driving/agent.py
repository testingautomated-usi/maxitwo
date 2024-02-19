from abc import ABC, abstractmethod
from typing import Dict

from custom_types import GymEnv

import numpy as np


class Agent(ABC):

    def __init__(self, env: GymEnv, env_name: str):
        self.env = env
        self.env_name = env_name

    @abstractmethod
    def predict(self, obs: np.ndarray, state: Dict) -> np.ndarray:
        raise NotImplementedError("Not implemented")
