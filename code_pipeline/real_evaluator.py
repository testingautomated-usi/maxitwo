import os.path
import time

import gym
import numpy as np

from code_pipeline.evaluator import Evaluator
from config import (
    DONKEY_SIM_NAME,
    BEAMNG_SIM_NAME,
    MAX_EPISODE_STEPS,
    UDACITY_SIM_NAME,
    MAX_EPISODE_STEPS_UDACITY,
)
from custom_types import GymEnv
from envs.beamng.config import MAP_SIZE
from factories import make_env, make_agent
from global_log import GlobalLog
from self_driving.agent import Agent
from test_generators.deepjanus_test_generator import JanusTestGenerator
from test_generators.mapelites.factories import make_features_given_feature_combination
from test_generators.mapelites.individual import Individual
from test_generators.mapelites.lateral_position_fitness import LateralPositionFitness

import platform

from utils.randomness import set_random_seed


class RealEvaluator(Evaluator):

    def __init__(
        self,
        env_name: str,
        env: GymEnv,
        agent: Agent,
        feature_combination: str,
        collect_images: bool = False,
    ):
        super(RealEvaluator, self).__init__(feature_combination=feature_combination)
        self.env_name = env_name
        self.env = env
        self.agent = agent
        self.logger = GlobalLog("real_evaluator")
        self.collect_images = collect_images

    def run_sim(self, individual: Individual) -> None:

        self.logger.info("Executing individual with id {}".format(individual.id))

        done = False
        obs = self.env.reset(individual=individual)
        steering_angles = []
        lateral_positions = []
        episode_length = 0
        state_dict = {}
        speeds = []
        observations = []

        while not done:
            action = self.agent.predict(obs=obs, state=state_dict)
            # Clip Action to avoid out of bound errors
            if isinstance(self.env.action_space, gym.spaces.Box):
                action = np.clip(
                    action, self.env.action_space.low, self.env.action_space.high
                )
            obs, done, info = self.env.step(action)

            state_dict["cte"] = info.get("cte", None)
            state_dict["cte_pid"] = info.get("cte_pid", None)
            state_dict["speed"] = info.get("speed", None)

            steering_angles.append(action[0])
            lateral_position = info.get("lateral_position", None)
            assert lateral_position is not None, "Lateral position needs to be present"
            lateral_positions.append(lateral_position)
            speeds.append(info.get("speed", None))

            episode_length += 1
            if self.collect_images:
                observations.append(obs)

            if episode_length > MAX_EPISODE_STEPS or (
                self.env_name == UDACITY_SIM_NAME
                and episode_length > MAX_EPISODE_STEPS_UDACITY
            ):
                self.logger.warn(
                    "Episode length {} > {}. Either the simulator went in background or "
                    "a communication error happened".format(
                        episode_length,
                        (
                            MAX_EPISODE_STEPS
                            if episode_length > MAX_EPISODE_STEPS
                            else MAX_EPISODE_STEPS_UDACITY
                        ),
                    )
                )
                raise RuntimeError()

            if done:

                self.logger.debug("Episode Length: {}".format(episode_length))
                self.logger.debug("Is success: {}".format(info["is_success"]))

                if episode_length <= 5:
                    # FIXME: for very short episodes (see Udacity where there is a bug that causes the CTE to be
                    #  very high at the beginning of the episodes) remove the actions and the observations from
                    #  the data and repeat the episode.
                    self.logger.warn("Very short episode: repeating...")
                    raise RuntimeError()

        # set fitness and all possible features
        individual.set_fitness(
            LateralPositionFitness(lateral_positions=lateral_positions)
        )

        individual.set_features(
            features=make_features_given_feature_combination(
                feature_combination=self.feature_combination,
                steering_angles=steering_angles,
                individual=individual,
            )
        )

        individual.set_behavioural_metrics(
            speeds=speeds,
            steering_angles=steering_angles,
            lateral_positions=lateral_positions,
        )

        individual.set_observations(observations=observations)

    def close(self) -> None:
        if self.env_name == BEAMNG_SIM_NAME:
            self.env.reset()
        else:
            self.env.reset(skip_generation=True)

        if self.env_name == DONKEY_SIM_NAME:
            time.sleep(2)
            self.env.exit_scene()
            self.env.close_connection()

        time.sleep(5)
        self.env.close()


if __name__ == "__main__":

    env_name = DONKEY_SIM_NAME
    seed = 0
    platform_ = platform.system()
    donkey_exe_path = "../../../Downloads/DonkeySimMacRepl/donkey_sim.app"
    udacity_exe_path = "../../../Downloads/UdacitySimMacRepl/udacity_sim.app"

    set_random_seed(seed=seed)

    assert platform_.lower() == "darwin", "Only on MacOS for now"
    assert os.path.exists(donkey_exe_path), "Donkey executor file not found: {}".format(
        donkey_exe_path
    )
    assert os.path.exists(
        udacity_exe_path
    ), "Udacity executor file not found: {}".format(udacity_exe_path)

    env = make_env(
        simulator_name=env_name, seed=seed, donkey_exe_path=donkey_exe_path, port=-1
    )

    test_generator = JanusTestGenerator(map_size=MAP_SIZE, simulator_name=env_name)
    # model_path = '../logs/models/mixed-dave2-2022_06_04_14_03_27.h5'  # robust model
    model_path = "../logs/models/mixed-dave2-2022_06_07_15_51_20.h5"  # weak model
    assert os.path.exists(model_path), "Model path not found: {}".format(model_path)

    agent = make_agent(
        env_name=env_name,
        env=env,
        model_path=model_path,
        agent_type="supervised",
        predict_throttle=False,
    )

    evaluator = RealEvaluator(env_name=env_name, env=env, agent=agent)

    for i in range(5):
        road = test_generator.generate()
        individual = Individual(road=road)
        evaluator.run_sim(individual=individual)
        print(
            "Fitness: {}. Features: {}".format(
                individual.get_fitness().get_value(),
                [feature.get_value() for feature in individual.get_features()],
            )
        )
        # Simulating computation time for computing next individual
        time.sleep(5)

    evaluator.close()
