import argparse
import logging
import os

import numpy as np

from config import AGENT_TYPES, MAX_ANGLE, MOCK_SIM_NAME, NUM_CONTROL_NODES, NUM_SAMPLED_POINTS, SIMULATOR_NAMES
from envs.beamng.config import MAP_SIZE
from factories import make_agent, make_env, make_test_generator
from global_log import GlobalLog
from test_generators.mapelites.config import CURVATURE_FEATURE_NAME, FEATURE_COMBINATIONS, TURNS_COUNT_FEATURE_NAME
from test_generators.mapelites.mapelites import MapElites
from utils.randomness import set_random_seed
from utils.report_utils import load_individual_report

parser = argparse.ArgumentParser()
parser.add_argument("--folder", help="Path of the folder where the logs are", type=str, default="logs")
parser.add_argument("--filepath", help="Paths of the folders where the reports are", type=str, required=True)
# run arguments
parser.add_argument("--env-name", help="Should be the name of the third simulator", type=str, choices=SIMULATOR_NAMES)
parser.add_argument("--donkey-exe-path", help="Path to the donkey simulator executor", type=str, default=None)
parser.add_argument("--udacity-exe-path", help="Path to the udacity simulator executor", type=str, default=None)
parser.add_argument("--beamng-user-path", help="Beamng user path", type=str, default=None)
parser.add_argument("--beamng-home-path", help="Beamng home path", type=str, default=None)
parser.add_argument("--seed", help="Random seed", type=int, default=-1)
parser.add_argument("--add-to-port", help="Modify default simulator port", type=int, default=-1)
parser.add_argument("--headless", help="Headless simulation", action="store_true", default=False)
parser.add_argument("--agent-type", help="Agent type", type=str, choices=AGENT_TYPES, default="random")
parser.add_argument("--test-generator", help="Which test generator to use", type=str, choices=["random"], default="random")
parser.add_argument(
    "--model-path", help="Path to agent model with extension (only if agent_type == 'supervised')", type=str, default=None
)
parser.add_argument(
    "--predict-throttle",
    help="Predict steering and throttle. Model to load must have been trained using an output dimension of 2",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--feature-combination",
    help="Feature combination",
    type=str,
    choices=FEATURE_COMBINATIONS,
    default="{}-{}".format(TURNS_COUNT_FEATURE_NAME, CURVATURE_FEATURE_NAME),
)
parser.add_argument("--collect-images", help="Collect images during execution", action="store_true", default=False)

args = parser.parse_args()

if __name__ == "__main__":

    logg = GlobalLog("run_individual_migration")

    # load probability map by default
    report = load_individual_report(filepath=os.path.join(args.folder, args.filepath))
    individuals = [individual for feature_bin in report.keys() for individual in report[feature_bin]]
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if args.seed == -1:
        args.seed = np.random.randint(2**30 - 1)

    set_random_seed(seed=args.seed)

    test_generator = make_test_generator(
        generator_name=args.test_generator,
        map_size=MAP_SIZE,
        simulator_name=args.env_name,
        agent_type=args.agent_type,
        num_control_nodes=NUM_CONTROL_NODES,
        max_angle=MAX_ANGLE,
        num_spline_nodes=NUM_SAMPLED_POINTS,
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
    )
    agent = make_agent(
        env_name=args.env_name,
        env=env,
        model_path=args.model_path,
        agent_type=args.agent_type,
        predict_throttle=args.predict_throttle,
    )

    logg.info("Disabling Shapely logs")
    for id in ["shapely.geos"]:
        l = logging.getLogger(id)
        l.setLevel(logging.CRITICAL)
        l.disabled = True

    mapelites = MapElites(
        env=env,
        env_name=args.env_name,
        agent=agent,
        filepath=args.folder,
        min_angle=0,
        max_angle=1,  # to pass the assertion
        mutation_extent=0,
        population_size=0,
        mock_evaluator=args.env_name == MOCK_SIM_NAME,
        iteration_runtime=0,
        test_generator=test_generator,
        individual_migration=True,
        feature_combination=args.feature_combination,
        port=args.add_to_port,
        donkey_exe_path=args.donkey_exe_path,
        udacity_exe_path=args.udacity_exe_path,
        beamng_home=args.beamng_home_path,
        beamng_user=args.beamng_user_path,
        headless=args.headless,
        collect_images=args.collect_images,
    )
    # load probability map by default
    mapelites.execute_individuals_and_place_in_map(individuals=individuals, occupation_map=False)
