import time
from typing import Dict

from config import DONKEY_SIM_NAME, BEAMNG_SIM_NAME, UDACITY_SIM_NAME
from custom_types import GymEnv

import numpy as np

from envs.donkey.config import KP_DONKEY, KD_DONKEY, KI_DONKEY
from envs.udacity.config import KP_UDACITY, KD_UDACITY, KI_UDACITY
from self_driving.agent import Agent


class AutopilotAgent(Agent):

    def __init__(self, env: GymEnv, env_name: str, max_speed: int, min_speed: int):
        super().__init__(env=env, env_name=env_name)

        self.previous_time = 0.0
        self.previous_cte = 0.0
        self.total_error = 0.0

        self.max_speed = max_speed
        self.min_speed = min_speed

    def predict(self, obs: np.ndarray, state: Dict) -> np.ndarray:
        if (
            self.env_name == DONKEY_SIM_NAME or self.env_name == UDACITY_SIM_NAME
        ) and len(state) > 0:

            delta = time.perf_counter() - self.previous_time

            diff_cte = (state["cte_pid"] - self.previous_cte) / delta
            self.previous_cte = state["cte_pid"]
            self.previous_time = time.perf_counter()

            self.total_error += state["cte_pid"]

            if self.env_name == DONKEY_SIM_NAME:
                steering = (
                    (-KP_DONKEY * state["cte_pid"])
                    - (KD_DONKEY * diff_cte)
                    - (KI_DONKEY * self.total_error)
                )
            elif self.env_name == UDACITY_SIM_NAME:
                steering = (
                    (-KP_UDACITY * state["cte_pid"])
                    - (KD_UDACITY * diff_cte)
                    - (KI_UDACITY * self.total_error)
                )
            else:
                raise RuntimeError("Unknown env name: {}".format(self.env_name))

            speed = state["speed"]
            if speed > self.max_speed:
                speed_limit = self.min_speed  # slow down
            else:
                speed_limit = self.max_speed

            throttle = np.clip(
                a=1.0 - steering**2 - (speed / speed_limit) ** 2, a_min=0.0, a_max=1.0
            )

            action = np.asarray([steering, throttle])

            return action

        if self.env_name == BEAMNG_SIM_NAME and len(state) > 0:
            return np.asarray([state["steering"], state["throttle"]])

        return self.env.action_space.sample()
