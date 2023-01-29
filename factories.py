import os.path
from typing import Tuple

from config import (
    AGENT_TYPES,
    BEAMNG_SIM_NAME,
    DONKEY_SIM_NAME,
    INPUT_SHAPE,
    MAX_ANGLE,
    MOCK_SIM_NAME,
    NUM_CONTROL_NODES,
    NUM_SAMPLED_POINTS,
    SIMULATOR_NAMES,
    UDACITY_SIM_NAME,
)
from custom_types import GymEnv
from cyclegan.models import create_model
from cyclegan.options.test_options import TestOptions
from cyclegan.util.util import get_base_and_test_default_options
from envs.beamng.beamng_gym_env import BeamngGymEnv
from envs.beamng.config import MAX_SPEED_BEAMNG, MIN_SPEED_BEAMNG
from envs.donkey.config import MAX_SPEED_DONKEY, MIN_SPEED_DONKEY
from envs.donkey.donkey_gym_env import DonkeyGymEnv
from envs.donkey.scenes.simulator_scenes import GeneratedTrack
from envs.mock_gym_env import MockGymEnv
from envs.udacity.config import MAX_SPEED_UDACITY, MIN_SPEED_UDACITY
from envs.udacity.udacity_gym_env import UdacityGymEnv
from self_driving.agent import Agent
from self_driving.autopilot_agent import AutopilotAgent
from self_driving.random_agent import RandomAgent
from self_driving.supervised_agent import SupervisedAgent
from test_generators.deepjanus_test_generator import JanusTestGenerator
from test_generators.sin_test_generator import SinTestGenerator
from test_generators.test_generator import TestGenerator


def make_test_generator(
    generator_name: str,
    map_size: int,
    simulator_name: str,
    agent_type: str,
    num_control_nodes: int = NUM_CONTROL_NODES,
    max_angle: int = MAX_ANGLE,
    num_spline_nodes: int = NUM_SAMPLED_POINTS,
) -> TestGenerator:
    if generator_name == "random":
        return JanusTestGenerator(
            map_size=map_size,
            simulator_name=simulator_name,
            agent_type=agent_type,
            num_control_nodes=num_control_nodes,
            max_angle=max_angle,
            num_spline_nodes=num_spline_nodes,
        )

    if generator_name == "sin":
        return SinTestGenerator(simulator_name=simulator_name)


def make_env(
    simulator_name: str,
    seed: int,
    port: int,
    test_generator: TestGenerator = None,
    donkey_exe_path: str = None,
    udacity_exe_path: str = None,
    headless: bool = False,
    beamng_user: str = None,
    beamng_home: str = None,
    beamng_autopilot: bool = False,
    cyclegan_experiment_name: str = None,
    gpu_ids: str = "-1",
    cyclegan_checkpoints_dir: str = None,
    cyclegan_epoch: int = None,
) -> GymEnv:
    assert simulator_name in SIMULATOR_NAMES, "Unknown simulator name {}. Choose among {}".format(
        simulator_name, SIMULATOR_NAMES
    )

    cyclegan_model = None
    cyclegan_options = None
    if cyclegan_experiment_name is not None:
        opt = get_base_and_test_default_options(
            name=cyclegan_experiment_name, gpu_ids=gpu_ids, checkpoints_dir=cyclegan_checkpoints_dir, epoch=cyclegan_epoch
        )

        cyclegan_model = create_model(opt)  # create a model given opt.model and other options
        cyclegan_model.setup(opt)  # regular setup: load and print networks; create schedulers
        cyclegan_options = opt

    if simulator_name == DONKEY_SIM_NAME:
        return DonkeyGymEnv(
            seed=seed,
            add_to_port=port,
            test_generator=test_generator,
            simulator_scene=GeneratedTrack(),
            headless=headless,
            exe_path=donkey_exe_path,
            cyclegan_model=cyclegan_model,
            cyclegan_options=cyclegan_options,
        )
    if simulator_name == BEAMNG_SIM_NAME:
        return BeamngGymEnv(
            seed=seed,
            add_to_port=port,
            test_generator=test_generator,
            beamng_user=beamng_user,
            beamng_home=beamng_home,
            autopilot=beamng_autopilot,
            cyclegan_model=cyclegan_model,
            cyclegan_options=cyclegan_options,
        )

    if simulator_name == UDACITY_SIM_NAME:
        return UdacityGymEnv(
            seed=seed,
            test_generator=test_generator,
            exe_path=udacity_exe_path,
            headless=headless,
            cyclegan_model=cyclegan_model,
            cyclegan_options=cyclegan_options,
        )

    if simulator_name == MOCK_SIM_NAME:
        return MockGymEnv()

    raise RuntimeError("Unknown simulator name: {}".format(simulator_name))


def get_max_min_speed(env_name: str) -> Tuple[int, int]:
    assert env_name in SIMULATOR_NAMES, "Unknown simulator name {}. Choose among {}".format(env_name, SIMULATOR_NAMES)

    if env_name == DONKEY_SIM_NAME:
        return MAX_SPEED_DONKEY, MIN_SPEED_DONKEY

    if env_name == UDACITY_SIM_NAME:
        return MAX_SPEED_UDACITY, MIN_SPEED_UDACITY

    if env_name == BEAMNG_SIM_NAME:
        return MAX_SPEED_BEAMNG, MIN_SPEED_BEAMNG

    if env_name == MOCK_SIM_NAME:
        return 30, 10  # completely random

    raise RuntimeError("Unknown simulator name: {}".format(env_name))


def make_agent(
    env_name: str, env: GymEnv, agent_type: str, model_path: str, predict_throttle: bool = False, fake_images: bool = False
) -> Agent:

    assert agent_type in AGENT_TYPES, "Unknown agent type {}. Choose among {}".format(agent_type, AGENT_TYPES)
    assert env_name in SIMULATOR_NAMES, "Unknown simulator name {}. Choose among {}".format(env_name, SIMULATOR_NAMES)

    max_speed, min_speed = get_max_min_speed(env_name=env_name)

    if agent_type == "supervised":
        assert os.path.exists(model_path), "Model path {} does not exist".format(model_path)
        return SupervisedAgent(
            env=env,
            env_name=env_name,
            max_speed=max_speed,
            min_speed=min_speed,
            model_path=model_path,
            input_shape=INPUT_SHAPE,
            predict_throttle=predict_throttle,
            fake_images=fake_images,
        )

    if agent_type == "autopilot":
        return AutopilotAgent(env=env, env_name=env_name, max_speed=max_speed, min_speed=min_speed)

    if agent_type == "random":
        return RandomAgent(env=env, env_name=env_name)

    raise RuntimeError("Unknown agent type: {}".format(agent_type))
