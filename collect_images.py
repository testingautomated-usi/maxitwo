import argparse
import datetime
import time

import gym
import numpy as np
from config import (
    SIMULATOR_NAMES,
    AGENT_TYPES,
    DONKEY_SIM_NAME,
    BEAMNG_SIM_NAME,
    TEST_GENERATORS,
    NUM_CONTROL_NODES,
    MAX_ANGLE,
    NUM_SAMPLED_POINTS,
)
from envs.beamng.config import MAP_SIZE
from factories import make_env, make_agent, make_test_generator
from global_log import GlobalLog
from utils.dataset_utils import save_archive
from utils.randomness import set_random_seed

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", help="Log folder", type=str, default="logs")
parser.add_argument(
    "--env-name", help="Env name", type=str, choices=SIMULATOR_NAMES, required=True
)
parser.add_argument(
    "--donkey-exe-path",
    help="Path to the donkey simulator executor",
    type=str,
    default=None,
)
parser.add_argument(
    "--udacity-exe-path",
    help="Path to the udacity simulator executor",
    type=str,
    default=None,
)
parser.add_argument(
    "--beamng-user-path", help="Beamng user path", type=str, default=None
)
parser.add_argument(
    "--beamng-home-path", help="Beamng home path", type=str, default=None
)
parser.add_argument("--seed", help="Random seed", type=int, default=-1)
parser.add_argument(
    "--add-to-port", help="Modify default simulator port", type=int, default=-1
)
parser.add_argument(
    "--num-episodes", help="Number of tracks to generate", type=int, default=3
)
parser.add_argument(
    "--headless", help="Headless simulation", action="store_true", default=False
)
parser.add_argument(
    "--no-save-archive",
    help="Disable archive storing",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--agent-type", help="Agent type", type=str, choices=AGENT_TYPES, default="random"
)
parser.add_argument(
    "--test-generator",
    help="Which test generator to use",
    type=str,
    choices=TEST_GENERATORS,
    default="random",
)
parser.add_argument(
    "--num-control-nodes",
    help="Number of control nodes of the generated road (only valid with random generator)",
    type=int,
    default=NUM_CONTROL_NODES,
)
parser.add_argument(
    "--max-angle",
    help="Max angle of a curve of the generated road (only valid with random generator)",
    type=int,
    default=MAX_ANGLE,
)
parser.add_argument(
    "--num-spline-nodes",
    help="Number of points to sample among control nodes of the generated road (only valid with random generator)",
    type=int,
    default=NUM_SAMPLED_POINTS,
)
parser.add_argument(
    "--model-path",
    help="Path to agent model with extension (only if agent_type == 'supervised')",
    type=str,
    default=None,
)
parser.add_argument(
    "--predict-throttle",
    help="Predict steering and throttle. Model to load must have been trained using an output dimension of 2",
    action="store_true",
    default=False,
)
# cyclegan options
parser.add_argument(
    "--cyclegan-experiment-name",
    type=str,
    default=None,
    help="name of the experiment. It decides where to store samples and models",
)
parser.add_argument(
    "--gpu-ids",
    type=str,
    default="-1",
    help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU",
)
parser.add_argument(
    "--cyclegan-checkpoints-dir", type=str, default=None, help="models are saved here"
)
parser.add_argument(
    "--cyclegan-epoch",
    type=str,
    default=-1,
    help="which epoch to load? set to latest to use latest cached model",
)

args = parser.parse_args()


if __name__ == "__main__":

    folder = args.folder
    logger = GlobalLog("collect_images")

    if args.seed == -1:
        args.seed = np.random.randint(2**32 - 1)

    set_random_seed(seed=args.seed)

    test_generator = make_test_generator(
        generator_name=args.test_generator,
        map_size=MAP_SIZE,
        simulator_name=args.env_name,
        agent_type=args.agent_type,
        num_control_nodes=args.num_control_nodes,
        max_angle=args.max_angle,
        num_spline_nodes=args.num_spline_nodes,
    )

    env = make_env(
        simulator_name=args.env_name,
        seed=args.seed,
        port=args.add_to_port,
        test_generator=test_generator,
        donkey_exe_path=args.donkey_exe_path,
        udacity_exe_path=args.udacity_exe_path,
        beamng_home=args.beamng_home_path,
        beamng_user=args.beamng_user_path,
        headless=args.headless,
        beamng_autopilot=args.agent_type == "autopilot",
        cyclegan_experiment_name=args.cyclegan_experiment_name,
        gpu_ids=args.gpu_ids,
        cyclegan_checkpoints_dir=args.cyclegan_checkpoints_dir,
        cyclegan_epoch=args.cyclegan_epoch,
    )
    agent = make_agent(
        env_name=args.env_name,
        env=env,
        model_path=args.model_path,
        agent_type=args.agent_type,
        predict_throttle=args.predict_throttle,
        fake_images=args.cyclegan_experiment_name is not None
        and args.cyclegan_checkpoints_dir is not None
        and args.cyclegan_epoch != -1,
    )

    actions = []
    observations = []
    tracks = []
    times_elapsed = []
    is_success_flags = []
    car_position_x_episodes = []
    car_position_y_episodes = []
    episode_lengths = []

    success_sum = 0

    episode_count = 0
    state_dict = dict()

    while episode_count < args.num_episodes:
        done, state = False, None
        episode_length = 0
        car_positions_x = []
        car_positions_y = []

        obs = env.reset()
        start_time = time.perf_counter()

        while not done:
            action = agent.predict(obs=obs, state=state_dict)
            # Clip Action to avoid out of bound errors
            if isinstance(env.action_space, gym.spaces.Box):
                action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, done, info = env.step(action)

            car_positions_x.append(info["pos"][0])
            car_positions_y.append(info["pos"][1])

            state_dict["cte"] = info.get("cte", None)
            state_dict["cte_pid"] = info.get("cte_pid", None)
            state_dict["speed"] = info.get("speed", None)
            lateral_position = info.get("lateral_position", None)

            state_dict["steering"] = info.get("steering", None)
            state_dict["throttle"] = info.get("throttle", None)

            # FIXME: harmonize the environments such that all have the same action space
            if args.env_name == BEAMNG_SIM_NAME and args.agent_type != "autopilot":
                assert (
                    info.get("throttle", None) is not None
                ), "Throttle is not defined for BeamNG"
                action = np.asarray([action[0], info.get("throttle")])

            # FIXME: first action is random for autopilots
            if episode_length > 0 and args.agent_type == "autopilot":
                actions.append(action)
                observations.append(obs)
            elif args.agent_type != "autopilot" and args.agent_type != "supervised":
                actions.append(action)
                observations.append(obs)
            elif args.agent_type == "supervised":
                actions.append(action)

            episode_length += 1

            if done:

                times_elapsed.append(time.perf_counter() - start_time)
                car_position_x_episodes.append(car_positions_x)
                car_position_y_episodes.append(car_positions_y)

                if info.get("track", None) is not None:
                    tracks.append(info["track"])

                if info.get("is_success", None) is not None:
                    success_sum += info["is_success"]
                    is_success_flags.append(info["is_success"])

                logger.debug("Episode #{}".format(episode_count + 1))
                logger.debug("Episode Length: {}".format(episode_length))
                logger.debug("Is success: {}".format(info["is_success"]))

                if episode_length <= 5:
                    # FIXME: for very short episodes (see Udacity where there is a bug that causes the CTE to be
                    #  very high at the beginning of the episodes) remove the actions and the observations from
                    #  the data and repeat the episode.
                    logger.warn("Removing short episode")
                    if args.agent_type == "autopilot":
                        original_length_actions = len(actions)
                        original_length_observations = len(observations)
                        items_to_remove = (
                            episode_length - 1
                            if args.agent_type == "autopilot"
                            else episode_length
                        )
                        # first random action of each episode is not included
                        condition = (
                            episode_length > 1
                            if args.agent_type == "autopilot"
                            else episode_length > 0
                        )
                        while condition:
                            actions.pop()
                            observations.pop()
                            episode_length -= 1
                            condition = (
                                episode_length > 1
                                if args.agent_type == "autopilot"
                                else episode_length > 0
                            )

                        assert (
                            len(actions) + items_to_remove == original_length_actions
                        ), "Error when removing actions. To remove: {}, Original: {}, New: {}".format(
                            items_to_remove, original_length_actions, len(actions)
                        )
                        assert (
                            len(observations) + items_to_remove
                            == original_length_observations
                        ), "Error when removing observations. To remove: {}, Original: {}, New: {}".format(
                            items_to_remove,
                            original_length_observations,
                            len(observations),
                        )
                    elif args.agent_type == "supervised":
                        original_length_actions = len(actions)
                        items_to_remove = episode_length
                        while episode_length > 0:
                            actions.pop()
                            observations.pop()
                            episode_length -= 1
                            condition = (
                                episode_length > 1
                                if args.agent_type == "autopilot"
                                else episode_length > 0
                            )

                        assert (
                            len(actions) + items_to_remove == original_length_actions
                        ), "Error when removing actions. To remove: {}, Original: {}, New: {}".format(
                            items_to_remove, original_length_actions, len(actions)
                        )

                    track_to_repeat = tracks.pop()
                    test_generator.set_road_to_generate(road=track_to_repeat)

                else:
                    episode_lengths.append(episode_length)
                    episode_count += 1

                state_dict = {}
                if args.no_save_archive:
                    actions.clear()
                    observations.clear()

    logger.debug("Success rate: {:.2f}".format(success_sum / episode_count))
    logger.debug("Mean time elapsed: {:.2f}s".format(np.mean(times_elapsed)))

    if not args.no_save_archive:
        save_archive(
            actions=actions,
            observations=observations,
            is_success_flags=is_success_flags,
            tracks=tracks,
            car_positions_x_episodes=car_position_x_episodes,
            car_positions_y_episodes=car_position_y_episodes,
            episode_lengths=episode_lengths,
            archive_path=folder,
            archive_name="{}-{}-archive-agent-{}-seed-{}-episodes-{}-max-angle-{}-length-{}".format(
                args.env_name,
                datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
                args.agent_type,
                args.seed,
                args.num_episodes,
                args.max_angle,
                args.num_control_nodes,
            ),
        )

    if args.env_name == BEAMNG_SIM_NAME:
        env.reset()
    else:
        env.reset(skip_generation=True)

    if args.env_name == DONKEY_SIM_NAME:
        time.sleep(2)
        env.exit_scene()
        env.close_connection()

    time.sleep(5)
    env.close()
