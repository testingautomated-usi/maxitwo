# Original author: Roma Sokolkov
# Edited by Antonin Raffin
from typing import NamedTuple

import gym
import numpy as np
from gym import spaces

import envs.cyclegan_wrapper
from config import BEAMNG_SIM_NAME
from envs.cyclegan_wrapper import CycleganWrapper
from custom_types import ObserveData
from cyclegan.models.test_model import TestModel
from envs.beamng.beamng_executor import BeamngExecutor
from envs.beamng.config import MIN_THROTTLE, MAX_THROTTLE, MAX_STEERING, INPUT_DIM
from global_log import GlobalLog
from test_generators.mapelites.individual import Individual
from test_generators.test_generator import TestGenerator


class BeamngGymEnv(gym.Env, CycleganWrapper):
    """
    Gym interface for BeamNG with support for using
    a VAE encoded observation instead of raw pixels if needed.
    """

    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    def __init__(
        self,
        seed: int,
        add_to_port: int,
        test_generator: TestGenerator,
        beamng_home: str,
        beamng_user: str,
        logs_folder_name: str = None,
        autopilot: bool = False,
        cyclegan_model: TestModel = None,
        cyclegan_options: NamedTuple = None,
    ):
        if cyclegan_model is not None:
            envs.cyclegan_wrapper.CycleganWrapper.__init__(
                self,
                env_name=BEAMNG_SIM_NAME,
                cyclegan_model=cyclegan_model,
                cyclegan_options=cyclegan_options,
            )

        self.min_throttle = MIN_THROTTLE
        self.max_throttle = MAX_THROTTLE
        self.test_generator = test_generator
        self.logger = GlobalLog("BeamngGymEnv")
        self.cyclegan_model = cyclegan_model
        self.cyclegan_options = cyclegan_options

        self.executor = BeamngExecutor(
            beamng_home=beamng_home,
            beamng_user=beamng_user,
            add_to_port=add_to_port,
            logs_folder_name=logs_folder_name,
            test_generator=self.test_generator,
            autopilot=autopilot,
        )

        self.action_space = spaces.Box(
            low=np.array([-MAX_STEERING]),
            high=np.array([MAX_STEERING]),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=0, high=255, shape=INPUT_DIM, dtype=np.uint8
        )

        self.seed(seed)
        self.count = 0

    def close_connection(self):
        return self.executor.close()

    def stop_simulation(self):
        # it seems that BeamNG naturally waits for the action to be returned by the agent
        pass
        # # it does not seem to work, vehicle continues to go
        # self.executor.pause()

    def restart_simulation(self):
        # it seems that BeamNG naturally waits for the action to be returned by the agent
        pass
        # # it does not seem to work, vehicle continues to go
        # self.executor.resume()

    def step(self, action: np.ndarray) -> ObserveData:
        """
        :param action: (np.ndarray)
        :return: (np.ndarray, bool, dict)
        """
        # action[0] is the steering angle

        self.executor.take_action(
            steering=action[0], throttle=action[1] if len(action) > 1 else None
        )
        observe_data = self.observe()

        return observe_data

    def reset(self, skip_generation: bool = False, individual: Individual = None):

        self.executor.reset(skip_generation=skip_generation, individual=individual)
        observation, done, info = self.observe()

        return observation

    def render(self, mode="human"):
        """
        :param mode: (str)
        """
        if mode == "rgb_array":
            return self.executor.original_image
        return None

    def observe(self) -> ObserveData:
        """
        :return: (np.ndarray, bool, dict)
        """
        observation, done, info = self.executor.observe()

        if self.cyclegan_model is not None:
            # im = self.get_fake_image(obs=observation)
            # fake = Image.fromarray(im)
            # original = Image.fromarray(observation)
            # original.show()
            # fake.show()
            return self.get_fake_image(obs=observation), done, info

        return observation, done, info

    def close(self):
        self.executor.close()

    def seed(self, seed=None):
        pass
