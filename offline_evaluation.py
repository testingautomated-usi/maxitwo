import argparse
import os

import numpy as np

from config import AGENT_TYPE_AUTOPILOT, AGENT_TYPES, SIMULATOR_NAMES
from factories import make_agent
from global_log import GlobalLog
from utils.dataset_utils import load_archive
from utils.randomness import set_random_seed

parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="Random seed", type=int, default=-1)
parser.add_argument("--agent-type", help="Agent type", type=str, choices=AGENT_TYPES, default="random")
parser.add_argument("--env-name", help="Should be the name of the third simulator", type=str, choices=SIMULATOR_NAMES)
parser.add_argument(
    "--model-path", help="Path to agent model with extension (only if agent_type == 'supervised')", type=str, default=None
)
parser.add_argument(
    "--predict-throttle",
    help="Predict steering and throttle. Model to load must have been trained using an output dimension of 2",
    action="store_true",
    default=False,
)
parser.add_argument("--archive-path", help="Archive path", type=str, default="logs")
parser.add_argument("--archive-name", help="Archive name to analyze", type=str, required=True)
args = parser.parse_args()


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    logger = GlobalLog("offline_evaluation")

    if args.seed == -1:
        args.seed = np.random.randint(2**32 - 1)

    set_random_seed(seed=args.seed)

    agent = make_agent(
        env_name=args.env_name,
        env=None,
        model_path=args.model_path,
        agent_type=args.agent_type,
        predict_throttle=args.predict_throttle,
        fake_images=args.model_path is not None and "fake" in args.model_path,
    )

    numpy_dict = load_archive(archive_path=args.archive_path, archive_name=args.archive_name)
    observations = numpy_dict["observations"]
    actions = numpy_dict["actions"]

    if len(actions.shape) > 2:
        actions = actions.squeeze(axis=1)

    if "episode_lengths" in numpy_dict:
        episode_lengths = numpy_dict["episode_lengths"]
        if AGENT_TYPE_AUTOPILOT in args.archive_name:
            # TODO the first action is random when the images are collected using the autopilot. Therefore,
            #  during collection the first image is skipped (i.e. not stored) but the episode length is incremented
            episode_lengths = [episode_length - 1 for episode_length in episode_lengths]

    prediction_errors = []
    all_prediction_errors = []
    previous_episode_lengths = 0
    total_prediction_errors = 0
    num_episodes = 0
    for i, observation in enumerate(observations):
        prediction = agent.predict(obs=observation, state={})

        if not args.predict_throttle:
            prediction = prediction[0]
        ground_truth = actions[i]
        if not args.predict_throttle:
            ground_truth = ground_truth[0]

        logger.info("Processing observation {}/{}".format(i + 1, len(observations)))
        if "episode_lengths" in numpy_dict:
            prediction_errors.append(np.abs(ground_truth - prediction))
            if i == previous_episode_lengths + (episode_lengths[num_episodes] - 1):
                assert episode_lengths[num_episodes] == len(
                    prediction_errors
                ), "Episode length {} != Num prediction errors {} for episode {}".format(
                    episode_lengths[num_episodes], len(prediction_errors), num_episodes
                )
                logger.info(
                    "======== Num episodes: {}. Length: {}, Prediction errors length: {} ======== ".format(
                        num_episodes,
                        episode_lengths[num_episodes],
                        len(prediction_errors),
                    )
                )
                previous_episode_lengths += episode_lengths[num_episodes]
                num_episodes += 1
                total_prediction_errors += len(prediction_errors)
                all_prediction_errors.append(np.asarray(prediction_errors))
                prediction_errors.clear()
        else:
            all_prediction_errors.append(np.abs(ground_truth - prediction))

    if "episode_lengths" in numpy_dict:
        assert len(observations) == total_prediction_errors, "Num observations {} != Total prediction errors {}".format(
            len(observations), total_prediction_errors
        )

    filename = (
        "offline-evaluation-fake-{}.npz".format(args.env_name)
        if "fake" in args.model_path
        else "offline-evaluation-{}.npz".format(args.env_name)
    )

    if "tracks_control_points" in numpy_dict:
        tracks_control_points = numpy_dict["tracks_control_points"]
        new_numpy_dict = {
            "tracks_control_points": tracks_control_points,
            "prediction_errors": np.asarray(all_prediction_errors),
        }
        logger.debug("Tracks control points shape: {}".format(new_numpy_dict["tracks_control_points"].shape))
    else:
        new_numpy_dict = {"prediction_errors": np.asarray(all_prediction_errors)}

    logger.debug("Prediction errors shape: {}".format(new_numpy_dict["prediction_errors"].shape))

    np.savez(os.path.join(args.archive_path, filename), **new_numpy_dict)
