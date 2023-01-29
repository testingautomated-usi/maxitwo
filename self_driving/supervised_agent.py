from typing import Dict, Tuple

import numpy as np

from autopilot_model import AutopilotModel
from custom_types import GymEnv
from global_log import GlobalLog
from self_driving.agent import Agent
from utils.dataset_utils import preprocess


class SupervisedAgent(Agent):
    def __init__(
        self,
        env: GymEnv,
        env_name: str,
        model_path: str,
        max_speed: int,
        min_speed: int,
        input_shape: Tuple[int],
        predict_throttle: bool = False,
        fake_images: bool = False,
    ):
        super().__init__(env=env, env_name=env_name)

        self.logger = GlobalLog("supevised_agent")

        self.agent = AutopilotModel(env_name=env_name, input_shape=input_shape, predict_throttle=predict_throttle)
        self.agent.load(model_path=model_path)
        self.predict_throttle = predict_throttle
        self.model_path = model_path
        self.fake_images = fake_images

        self.max_speed = max_speed
        self.min_speed = min_speed

    def predict(self, obs: np.ndarray, state: Dict) -> np.ndarray:

        obs = preprocess(image=obs, env_name=self.env_name, fake_images=self.fake_images)

        # the model expects 4D array
        obs = np.array([obs])

        if self.predict_throttle:
            action = self.agent.model.predict(obs, batch_size=1)
            steering, throttle = action[0], action[1]
        else:
            steering = float(self.agent.model.predict(obs, batch_size=1))

            speed = 0.0 if state.get("speed", None) is None else state["speed"]

            if speed > self.max_speed:
                speed_limit = self.min_speed  # slow down
            else:
                speed_limit = self.max_speed

            # steering = self.change_steering(steering=steering)
            throttle = np.clip(a=1.0 - steering**2 - (speed / speed_limit) ** 2, a_min=0.0, a_max=1.0)

            # if the track begins with a curve the model steers at the maximum and the throttle will be 0 since the
            # speed is 0. To counteract this give a non-zero throttle such that the car can start going
            if abs(steering) >= 1.0 and throttle == 0.0 and speed == 0.0 and len(state) > 0:
                self.logger.warn("Road starts with a curve! Giving the car an extra throttle")
                throttle = 0.5

        return np.asarray([steering, throttle])
