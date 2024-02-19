# Original author: Roma Sokolkov
# Edited by Antonin Raffin
import os
import time
from typing import NamedTuple

import gym
import numpy as np
from gym import spaces

import envs.cyclegan_wrapper
from envs.cyclegan_wrapper import CycleganWrapper
from config import UDACITY_SIM_NAME
from cyclegan.models.test_model import TestModel
from envs.udacity.config import BASE_PORT, MAX_STEERING, INPUT_DIM
from envs.udacity.core.udacity_sim import UdacitySimController
from envs.unity_proc import UnityProcess
from global_log import GlobalLog
from test_generators.mapelites.individual import Individual
from test_generators.test_generator import TestGenerator
from custom_types import ObserveData


class UdacityGymEnv(gym.Env, CycleganWrapper):
    """
    Gym interface for DonkeyCar with support for using
    a VAE encoded observation instead of raw pixels if needed.
    """

    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    def __init__(
        self,
        seed: int,
        test_generator: TestGenerator = None,
        headless: bool = False,
        exe_path: str = None,
        cyclegan_model: TestModel = None,
        cyclegan_options: NamedTuple = None,
    ):

        if cyclegan_model is not None:
            envs.cyclegan_wrapper.CycleganWrapper.__init__(
                self,
                env_name=UDACITY_SIM_NAME,
                cyclegan_model=cyclegan_model,
                cyclegan_options=cyclegan_options,
            )

        self.seed = seed
        self.exe_path = exe_path
        self.logger = GlobalLog("UdacityGymEnv")
        self.test_generator = test_generator
        if headless:
            self.logger.warn("Headless mode not supported with Udacity")
        self.headless = False
        self.port = BASE_PORT
        self.cyclegan_model = cyclegan_model
        self.cyclegan_options = cyclegan_options

        self.unity_process = None
        if self.exe_path is not None:
            self.logger.info("Starting UdacityGym env")
            assert os.path.exists(self.exe_path), "Path {} does not exist".format(
                self.exe_path
            )
            # Start Unity simulation subprocess if needed
            self.unity_process = UnityProcess(sim_name=UDACITY_SIM_NAME)
            self.unity_process.start(
                sim_path=self.exe_path, headless=headless, port=self.port
            )
            time.sleep(
                2
            )  # wait for the simulator to start and the scene to be selected

        self.executor = UdacitySimController(
            port=self.port, test_generator=test_generator
        )

        # steering + throttle, action space must be symmetric
        self.action_space = spaces.Box(
            low=np.array([-MAX_STEERING, -1]),
            high=np.array([MAX_STEERING, 1]),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=0, high=255, shape=INPUT_DIM, dtype=np.uint8
        )

    def stop_simulation(self):
        raise NotImplementedError("Not implemented")

    def restart_simulation(self):
        raise NotImplementedError("Not implemented")

    def step(self, action: np.ndarray) -> ObserveData:
        """
        :param action: (np.ndarray)
        :return: (np.ndarray, float, bool, dict)
        """
        # action[0] is the steering angle
        # action[1] is the throttle

        self.executor.take_action(action=action)
        observation, done, info = self.observe()

        return observation, done, info

    def reset(
        self, skip_generation: bool = False, individual: Individual = None
    ) -> np.ndarray:

        self.executor.reset(skip_generation=skip_generation, individual=individual)
        observation, done, info = self.observe()

        return observation

    def render(self, mode="human"):
        """
        :param mode: (str)
        """
        if mode == "rgb_array":
            return self.executor.image_array
        return None

    def observe(self) -> ObserveData:
        """
        Encode the observation using VAE if needed.

        :return: (np.ndarray, float, bool, dict)
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
        if self.unity_process is not None:
            self.unity_process.quit()
