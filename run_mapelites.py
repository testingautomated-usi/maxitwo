import argparse
import datetime
import logging

import numpy as np

from config import (
    SIMULATOR_NAMES,
    AGENT_TYPES,
    NUM_CONTROL_NODES,
    MAX_ANGLE,
    NUM_SAMPLED_POINTS,
    MIN_ANGLE,
    MOCK_SIM_NAME,
    BEAMNG_SIM_NAME,
)
from envs.beamng.config import MAP_SIZE
from factories import make_env, make_agent, make_test_generator
from global_log import GlobalLog
from test_generators.mapelites.config import (
    TURNS_COUNT_FEATURE_NAME,
    CURVATURE_FEATURE_NAME,
    FEATURE_COMBINATIONS,
)
from test_generators.mapelites.mapelites import MapElites
from utils.randomness import set_random_seed
from utils.report_utils import plot_and_save_probability_map

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
    "--min-angle",
    help="Min angle of a curve of the generated road (only valid with random generator)",
    type=int,
    default=MIN_ANGLE,
)
parser.add_argument(
    "--num-spline-nodes",
    help="Number of points to sample among control nodes of the generated road (only valid with random generator)",
    type=int,
    default=NUM_SAMPLED_POINTS,
)
parser.add_argument(
    "--add-to-port", help="Modify default simulator port", type=int, default=-1
)
parser.add_argument(
    "--headless", help="Headless simulation", action="store_true", default=False
)
parser.add_argument(
    "--agent-type", help="Agent type", type=str, choices=AGENT_TYPES, default="random"
)
parser.add_argument(
    "--test-generator",
    help="Which test generator to use",
    type=str,
    choices=["random"],
    default="random",
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
parser.add_argument(
    "--cyclegan",
    help="Whether to load the cyclegan",
    action="store_true",
    default=False,
)
# mapelites parameters
parser.add_argument(
    "--population-size", help="Size of the initial population", type=int, default=40
)
parser.add_argument(
    "--feature-combination",
    help="Feature combination",
    type=str,
    choices=FEATURE_COMBINATIONS,
    default="{}-{}".format(TURNS_COUNT_FEATURE_NAME, CURVATURE_FEATURE_NAME),
)
parser.add_argument(
    "--iteration-runtime",
    help="Search budget of the algorithm (in number of iterations)",
    type=int,
    default=60,
)
parser.add_argument(
    "--mutation-extent",
    help="How much each control point coordinate can be changed",
    type=int,
    default=6,
)
parser.add_argument(
    "--num-runs",
    help="Number of times the search should be executed. If > 0 then it also computes the probability map.",
    type=int,
    default=-1,
)
parser.add_argument(
    "--resume-datestr",
    help="Date string needed to resume previous execution.",
    type=str,
    default=None,
)
parser.add_argument("--resume-run", help="Executed runs.", type=int, default=-1)
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
    logger = GlobalLog("run_mapelites")

    if args.seed == -1:
        args.seed = np.random.randint(2**30 - 1)

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

    logger.info("Disabling Shapely logs")
    for id in ["shapely.geos"]:
        l = logging.getLogger(id)
        l.setLevel(logging.CRITICAL)
        l.disabled = True

    if args.num_runs == -1:
        mapelites = MapElites(
            env=env,
            env_name=args.env_name,
            agent=agent,
            filepath=folder,
            min_angle=args.min_angle,
            max_angle=args.max_angle,
            mutation_extent=args.mutation_extent,
            population_size=args.population_size,
            mock_evaluator=args.env_name == MOCK_SIM_NAME,
            iteration_runtime=args.iteration_runtime,
            test_generator=test_generator,
            seed=args.seed,
            port=args.add_to_port,
            donkey_exe_path=args.donkey_exe_path,
            udacity_exe_path=args.udacity_exe_path,
            beamng_home=args.beamng_home_path,
            beamng_user=args.beamng_user_path,
            headless=args.headless,
            beamng_autopilot=args.agent_type == "autopilot",
            feature_combination=args.feature_combination,
            cyclegan_experiment_name=args.cyclegan_experiment_name,
            cyclegan_epoch=args.cyclegan_epoch,
            cyclegan_checkpoints_dir=args.cyclegan_checkpoints_dir,
            gpu_ids=args.gpu_ids,
        )
        mapelites.run(last=True)
    else:
        if args.resume_datestr is not None:
            str_datetime = args.resume_datestr
        else:
            str_datetime = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        range_num_runs = (
            range(args.num_runs)
            if args.resume_run == -1
            else range(args.resume_run, args.num_runs)
        )

        for run_id in range_num_runs:
            logger.info("======== Mapelites run #{} ========".format(run_id))
            mapelites = MapElites(
                env=env,
                env_name=args.env_name,
                agent=agent,
                filepath=folder,
                min_angle=args.min_angle,
                max_angle=args.max_angle,
                mutation_extent=args.mutation_extent,
                population_size=args.population_size,
                mock_evaluator=args.env_name == MOCK_SIM_NAME,
                iteration_runtime=args.iteration_runtime,
                test_generator=test_generator,
                seed=args.seed,
                port=args.add_to_port,
                donkey_exe_path=args.donkey_exe_path,
                udacity_exe_path=args.udacity_exe_path,
                beamng_home=args.beamng_home_path,
                beamng_user=args.beamng_user_path,
                headless=args.headless,
                beamng_autopilot=args.agent_type == "autopilot",
                run_id=run_id,
                str_datetime=str_datetime,
                feature_combination=args.feature_combination,
                cyclegan_experiment_name=args.cyclegan_experiment_name,
                cyclegan_epoch=args.cyclegan_epoch,
                cyclegan_checkpoints_dir=args.cyclegan_checkpoints_dir,
                gpu_ids=args.gpu_ids,
            )
            if run_id == args.num_runs - 1:
                mapelites.run(last=True)
            else:
                mapelites.run()

            if run_id != args.num_runs - 1 and args.env_name == BEAMNG_SIM_NAME:
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
                    cyclegan_epoch=args.cyclegan_epoch,
                    cyclegan_checkpoints_dir=args.cyclegan_checkpoints_dir,
                    gpu_ids=args.gpu_ids,
                )

        plot_and_save_probability_map(
            report_name_suffix="mapelites",
            filepath=args.folder,
            env_name=args.env_name,
            str_datetime=str_datetime,
            num_runs=args.num_runs,
        )
