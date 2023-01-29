import argparse
import glob
import logging
import os
from typing import List

import numpy as np

from config import AGENT_TYPES, MAX_ANGLE, MIN_ANGLE, MOCK_SIM_NAME, NUM_CONTROL_NODES, NUM_SAMPLED_POINTS, SIMULATOR_NAMES
from envs.beamng.config import MAP_SIZE
from factories import make_agent, make_env, make_test_generator
from global_log import GlobalLog
from test_generators.mapelites.config import (
    CURVATURE_FEATURE_NAME,
    FEATURE_COMBINATIONS,
    QUALITY_METRICS_NAMES,
    TURNS_COUNT_FEATURE_NAME,
)
from test_generators.mapelites.individual import Individual
from test_generators.mapelites.mapelites import MapElites
from utils.randomness import set_random_seed
from utils.report_utils import (
    get_name_min_and_max_2d_features,
    load_individual_report,
    load_mapelites_report,
    plot_map_of_elites,
    plot_raw_map_of_elites,
    resize_map_of_elites,
    write_mapelites_report,
)

parser = argparse.ArgumentParser()
parser.add_argument("--folder", help="Path of the folder where the logs are", type=str, default="logs")
parser.add_argument("--filepaths", nargs="+", help="Paths of the folders where the reports are", type=str, required=True)
parser.add_argument("--output-dir", help="Output folder where the merged heatmap will be saved", type=str, default=None)
parser.add_argument("--execute", help="Run all individuals in the merged population", action="store_true", default=False)
parser.add_argument(
    "--quality-metric", help="Name of the quality metric", type=str, choices=QUALITY_METRICS_NAMES, default=None
)
parser.add_argument(
    "--min-quality-metric",
    help="Min value of the quality metric specified above (for normalization purposes)",
    type=float,
    default=None,
)
parser.add_argument(
    "--max-quality-metric",
    help="Max value of the quality metric specified above (for normalization purposes)",
    type=float,
    default=None,
)
parser.add_argument(
    "--quality-metric-merge",
    help="How to merge the quality metric maps",
    type=str,
    choices=["avg", "min", "max"],
    default=None,
)
parser.add_argument("--load-probability-map", help="Load probability map", action="store_true", default=False)
parser.add_argument(
    "--multiply-probabilities",
    help="Multiply probabilities when merging the probability maps (by default the probabilities are averaged)",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--weighted-average-probabilities",
    help="Whether to consider a weighted average of the probabilities",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--failure-probability",
    help="Whether to consider failure probability when building the map (default success_probability)",
    action="store_true",
    default=False,
)
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

args = parser.parse_args()

# FIXME: Merge multiple maps of the across different simulators: consider refactoring with merge_maps_simulator.py
#  this file loads the reports, computes the metrics from scratch and merges those
if __name__ == "__main__":

    assert len(args.filepaths) <= 2, "Cannot merge more than 2 maps"

    for filepath in args.filepaths:
        assert os.path.exists(os.path.join(args.folder, filepath)), "{} does not exist".format(
            os.path.join(args.folder, filepath)
        )

    if args.output_dir is None:
        merged_heatmap_filepath = os.path.join(args.folder, "merged_{}".format("_".join(args.filepaths)))
    else:
        merged_heatmap_filepath = os.path.join(args.folder, args.output_dir)

    os.makedirs(name=merged_heatmap_filepath, exist_ok=True)

    logg = GlobalLog("merge_mapelites")

    reports = []

    for filepath in args.filepaths:
        if args.load_probability_map or args.quality_metric is not None:
            report_filepath = os.path.join(args.folder, filepath)
            assert os.path.exists(report_filepath), "Report file {} does not exist".format(report_filepath)
            reports.append(load_individual_report(filepath=report_filepath))
        else:
            all_report_files = glob.glob(os.path.join(args.folder, filepath, "report_iterations_*.json"))
            report_file = list(filter(lambda rf: int(rf[rf.rindex("_") + 1 : rf.rindex(".")]) > 0, all_report_files))
            assert len(report_file) == 1, "Only one match supported. Found: {}".format(len(report_file))
            report_file = report_file[0]
            reports.append(load_mapelites_report(filepath=report_file))

    individuals_in_population: List[Individual] = []
    for i, report in enumerate(reports):
        if args.load_probability_map or args.quality_metric is not None:
            all_individuals_report = [individual for feature_bin in report.keys() for individual in report[feature_bin]]
            individuals_in_population.extend(all_individuals_report)
            logg.info("Report #{}, all individuals report: {}".format(i, len(all_individuals_report)))
        else:
            individuals_in_population.extend(
                list(filter(lambda ind: ind.id in report["ids_in_population"], report["individuals"]))
            )
            logg.info(
                "Report #{}, all individuals length: {}, individuals in population: {}".format(
                    i, len(report["individuals"]), len(report["ids_in_population"])
                )
            )

    (
        feature_x_name,
        feature_y_name,
        min_feature_x,
        max_feature_x,
        min_feature_y,
        max_feature_y,
    ) = get_name_min_and_max_2d_features(individuals=individuals_in_population)

    logg.info(
        "Feature x: {} in [{}, {}], Feature y: {} in [{}, {}]".format(
            feature_x_name, min_feature_x, max_feature_x, feature_y_name, min_feature_y, max_feature_y
        )
    )

    resized_maps = []
    resized_maps_counts = []
    fill_value = None
    for i, report in enumerate(reports):
        if args.load_probability_map or args.quality_metric is not None:
            individuals = [individual for feature_bin in report.keys() for individual in report[feature_bin]]
        else:
            individuals = list(filter(lambda ind: ind.id in report["ids_in_population"], report["individuals"]))

        resized_map, resized_map_counts, fill_value = resize_map_of_elites(
            x_axis_min=min_feature_x,
            x_axis_max=max_feature_x,
            y_axis_min=min_feature_y,
            y_axis_max=max_feature_y,
            individuals=individuals,
            occupation_map=not args.failure_probability,
            failure_probability=args.failure_probability,
            quality_metric=args.quality_metric,
        )

        resized_maps.append(resized_map)
        resized_maps_counts.append(resized_map_counts)

    # assuming two maps
    assert len(resized_maps) == 2, "Only two maps are supported at the moment"

    resized_map_1 = resized_maps[0]
    resized_map_2 = resized_maps[1]

    resized_map_counts_1 = resized_maps_counts[0]
    resized_map_counts_2 = resized_maps_counts[1]

    valued_keys_1 = set(filter(lambda key: resized_map_1[key] != fill_value, resized_map_1.keys()))
    valued_keys_2 = set(filter(lambda key: resized_map_2[key] != fill_value, resized_map_2.keys()))

    bins_intersection = valued_keys_1.intersection(valued_keys_2)
    logg.info("# Keys that conflict: {}".format(len(bins_intersection)))

    values = dict()

    if args.load_probability_map or args.quality_metric is not None:
        # resized_map_1 and resized_map_2 have the same keys
        for k in resized_map_1.keys():
            if k in bins_intersection:
                value_1 = resized_map_1[k]
                value_2 = resized_map_2[k]

                if args.load_probability_map:
                    if args.multiply_probabilities:
                        values[k] = value_1 * value_2
                    else:
                        values[k] = (value_1 + value_2) / 2
                elif args.quality_metric is not None:
                    assert args.min_quality_metric is not None, "min_quality_metric argument is needed for normalization"
                    assert args.max_quality_metric is not None, "max_quality_metric argument is needed for normalization"
                    assert (
                        args.min_quality_metric < args.max_quality_metric
                    ), "Min quality metric {} > Max quality metric {}".format(args.min_quality_metric, args.max_quality_metric)
                    normalized_value_1 = (value_1 - args.min_quality_metric) / (
                        args.max_quality_metric - args.min_quality_metric
                    )
                    normalized_value_2 = (value_2 - args.min_quality_metric) / (
                        args.max_quality_metric - args.min_quality_metric
                    )

                    assert (
                        0 <= normalized_value_1 <= 1
                    ), "Value {}, original {}, not in bounds, index: {}, merge type: {}, bounds: ({}, {})".format(
                        normalized_value_1,
                        value_1,
                        k,
                        args.quality_metric_merge,
                        args.min_quality_metric,
                        args.max_quality_metric,
                    )
                    assert (
                        0 <= normalized_value_2 <= 1
                    ), "Value {}, original {}, not in bounds, index: {}, merge type: {}, bounds: ({}, {})".format(
                        normalized_value_2,
                        value_2,
                        k,
                        args.quality_metric_merge,
                        args.min_quality_metric,
                        args.max_quality_metric,
                    )

                    if args.quality_metric_merge == "avg":
                        values[k] = (normalized_value_1 + normalized_value_2) / 2
                    elif args.quality_metric_merge == "min":
                        values[k] = min(normalized_value_1, normalized_value_2)
                    elif args.quality_metric_merge == "max":
                        values[k] = max(normalized_value_1, normalized_value_2)
                    else:
                        raise RuntimeError("Unknown quality_metric_merge: {}".format(args.quality_metric_merge))

                    assert 0 <= values[k] <= 1, "Value {} not in bounds, index: {}, merge type: {}".format(
                        values[k], k, args.quality_metric_merge
                    )

                logg.info(
                    "Conflict of key {} between the two maps: {} vs {}. Value in map: {}".format(
                        k, resized_map_1[k], resized_map_2[k], values[k]
                    )
                )

            elif resized_map_1[k] != fill_value:
                raise NotImplementedError("First map has a key {} different from fill value {}".format(k, fill_value))
            elif resized_map_2[k] != fill_value:
                raise NotImplementedError("Second map has a key {} different from fill value {}".format(k, fill_value))
            else:
                values[k] = fill_value
    else:
        fitness_values = dict()
        # resized_map_1 and resized_map_2 have the same keys
        for k in resized_map_1.keys():
            if k in bins_intersection:
                logg.info("Conflict of keys between the two maps: {} vs {}".format(resized_map_1[k], resized_map_2[k]))
                if resized_map_1[k] < resized_map_2[k]:
                    fitness_values[k] = resized_map_1[k]
                else:
                    fitness_values[k] = resized_map_2[k]

                if args.failure_probability:
                    value_1 = 1.0 if resized_map_1[k] < 0.0 else 0.0
                    value_2 = 1.0 if resized_map_2[k] < 0.0 else 0.0
                else:
                    value_1 = 1.0 if resized_map_1[k] > 0.0 else 0.0
                    value_2 = 1.0 if resized_map_2[k] > 0.0 else 0.0

                values[k] = (value_1 + value_2) / 2
            elif resized_map_1[k] != fill_value:
                fitness_values[k] = resized_map_1[k]
                if args.failure_probability:
                    values[k] = 1.0 if resized_map_1[k] < 0.0 else 0.0
                else:
                    values[k] = 1.0 if resized_map_1[k] > 0.0 else 0.0
            elif resized_map_2[k] != fill_value:
                fitness_values[k] = resized_map_2[k]
                if args.failure_probability:
                    values[k] = 1.0 if resized_map_2[k] < 0.0 else 0.0
                else:
                    values[k] = 1.0 if resized_map_2[k] > 0.0 else 0.0
            else:
                fitness_values[k] = fill_value
                values[k] = fill_value

        write_mapelites_report(
            filepath=merged_heatmap_filepath,
            iterations=0,
            population=None,
            fitness_values=fitness_values.values(),
            individuals=individuals_in_population,
        )

        plot_map_of_elites(
            data=fitness_values,
            filepath=merged_heatmap_filepath,
            iterations=0,
            x_axis_label=feature_x_name,
            y_axis_label=feature_y_name,
            min_value_cbar=individuals_in_population[0].get_fitness().get_min_value(),
            max_value_cbar=individuals_in_population[0].get_fitness().get_max_value(),
            occupation_map=True,
        )

        plot_raw_map_of_elites(
            data=fitness_values,
            filepath=merged_heatmap_filepath,
            iterations=0,
            x_axis_label=feature_x_name,
            y_axis_label=feature_y_name,
            occupation_map=True,
        )

    plot_map_of_elites(
        data=values,
        filepath=merged_heatmap_filepath,
        iterations=0,
        x_axis_label=feature_x_name,
        y_axis_label=feature_y_name,
        min_value_cbar=0.0,
        max_value_cbar=1.0,
        occupation_map=False,
        multiply_probabilities=args.multiply_probabilities,
        failure_probability=args.failure_probability,
        quality_metric=args.quality_metric,
        quality_metric_merge=args.quality_metric_merge,
    )

    plot_raw_map_of_elites(
        data=values,
        filepath=merged_heatmap_filepath,
        iterations=0,
        x_axis_label=feature_x_name,
        y_axis_label=feature_y_name,
        occupation_map=False,
        multiply_probabilities=args.multiply_probabilities,
        failure_probability=args.failure_probability,
        quality_metric_merge=args.quality_metric_merge,
        quality_metric=args.quality_metric,
    )

    if args.execute:

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
            merged_heatmap=True,
            feature_combination=args.feature_combination,
        )
        mapelites.execute_individuals_and_place_in_map(individuals=individuals_in_population)
