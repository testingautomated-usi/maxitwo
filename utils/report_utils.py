import glob
import json
import os
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import Point

from config import SIMULATOR_NAMES
from global_log import GlobalLog
from self_driving.road_utils import get_road
from test_generators.mapelites.config import (
    QUALITY_METRICS_NAMES,
    CURVATURE_FEATURE_NAME,
)
from test_generators.mapelites.factories import (
    make_features_given_concrete_features,
    make_quality_metric,
    make_behavioural_metrics_given_dict,
)
from test_generators.mapelites.individual import Individual
from test_generators.mapelites.lateral_position_fitness import LateralPositionFitness

FILL_VALUE = np.inf


def load_mapelites_report(filepath: str) -> Dict:

    logg = GlobalLog("load_mapelites_report")
    assert os.path.exists(filepath), "{} does not exist".format(filepath)
    assert ".json" in filepath, "Filename {} is not a json file".format(filepath)
    logg.info("Loading {}".format(filepath))
    with open(filepath, "r") as f:
        report = json.loads(f.read())

    ids_in_population = report["individuals_in_population"]

    simulator_name = None
    for sim_name in SIMULATOR_NAMES:
        if sim_name in filepath:
            simulator_name = sim_name
            break
    assert (
        simulator_name is not None
    ), "Simulator name {} not found in filepath {}".format(SIMULATOR_NAMES, filepath)

    individuals = []
    for individual_dict in report["individuals"]:
        individual = _make_individual_from_report(
            filepath=filepath,
            individual_dict=individual_dict,
            simulator_name=simulator_name,
        )
        individuals.append(individual)

    result = dict()
    result["ids_in_population"] = ids_in_population
    result["individuals"] = individuals

    return result


def _make_individual_from_report(
    filepath: str, individual_dict: Dict, simulator_name: str
) -> Individual:
    individual_id = individual_dict["id"]
    individual_features = individual_dict["features"]
    individual_fitness = individual_dict["fitness"]
    individual_representation = individual_dict["representation"]

    features = make_features_given_concrete_features(
        concrete_features=individual_features
    )

    # FIXME: replace with fitness factory
    if type(individual_fitness) == tuple or isinstance(individual_fitness, list):
        assert (
            individual_fitness[0] == "lateral_position_fitness"
        ), "Unknown fitness type: {}".format(individual_fitness[0])
        fitness = LateralPositionFitness(
            lateral_positions=[], min_lateral_position=individual_fitness[1]
        )
    elif isinstance(individual_fitness, float):
        fitness = LateralPositionFitness(
            lateral_positions=[], min_lateral_position=individual_fitness
        )
    else:
        raise RuntimeError(
            "Individual fitness type {} not supported. Report: {}".format(
                individual_fitness, filepath
            )
        )

    control_points = [
        Point(cp[0], cp[1], cp[2]) for cp in individual_representation["control_points"]
    ]
    road_points = [
        Point(rp[0], rp[1]) for rp in individual_representation["road_points"]
    ]
    road_width = individual_representation["road_width"]

    road = get_road(
        control_points=control_points,
        road_points=road_points,
        road_width=road_width,
        simulator_name=simulator_name,
    )

    behavioural_metrics = make_behavioural_metrics_given_dict(
        individual_dict=individual_dict
    )
    individual = Individual(road=road)
    individual.set_features(features=tuple(features))
    individual.set_fitness(fitness=fitness)
    assert len(behavioural_metrics) > 0, "No behavioural metrics in {} for {}".format(
        filepath, simulator_name
    )
    individual.set_behavioural_metrics_dict(behavioural_metrics=behavioural_metrics)

    individual.id = individual_id

    return individual


def load_raw_map(
    filepath: str,
    failure_probability: bool,
    multiply_probabilities: bool,
    quality_metric: str = None,
    quality_metric_merge: str = None,
    weighted_average_probabilities: bool = False,
    full_filepath: str = None,
) -> Dict:

    logg = GlobalLog("load_raw_map")

    if full_filepath is None:
        if quality_metric is None:
            failure_label = "failure" if failure_probability else "success"
            # removing prefix when checking which probability map to load
            if multiply_probabilities and (
                filepath[filepath.rindex(os.path.sep) + 1 :].startswith(
                    "merged_mapelites"
                )
                or filepath[filepath.rindex(os.path.sep) + 1 :].startswith(
                    "merged_merged"
                )
            ):
                report_files = glob.glob(
                    os.path.join(
                        filepath,
                        "raw_heatmap_{}_probability_multiply_*.json".format(
                            failure_label
                        ),
                    )
                )
            elif weighted_average_probabilities and (
                filepath[filepath.rindex(os.path.sep) + 1 :].startswith(
                    "merged_mapelites"
                )
                or filepath[filepath.rindex(os.path.sep) + 1 :].startswith(
                    "merged_merged"
                )
            ):
                report_files = glob.glob(
                    os.path.join(
                        filepath,
                        "raw_heatmap_{}_probability_weighted_*.json".format(
                            failure_label
                        ),
                    )
                )
            else:
                report_files = glob.glob(
                    os.path.join(
                        filepath,
                        "raw_heatmap_{}_probability_*.json".format(failure_label),
                    )
                )

            if len(report_files) == 0:
                logg.warn("Failed to load map, trying to load the standard map")
                report_files = glob.glob(
                    os.path.join(
                        filepath,
                        "raw_heatmap_{}_probability_*.json".format(failure_label),
                    )
                )

        else:
            if quality_metric_merge is not None:
                report_files = glob.glob(
                    os.path.join(
                        filepath,
                        "raw_heatmap_{}_{}_*.json".format(
                            quality_metric, quality_metric_merge
                        ),
                    )
                )
            else:
                report_files = glob.glob(
                    os.path.join(
                        filepath, "raw_heatmap_{}_*.json".format(quality_metric)
                    )
                )

        assert len(report_files), "Only one match supported, found: {}".format(
            len(report_files)
        )

        report_file = report_files[0]
    else:
        report_file = full_filepath

    logg.info("Loading {}".format(report_file))
    with open(report_file, "r") as f:
        report = json.loads(f.read())

    report_dict = dict()
    for feature_bin_str, value in report.items():
        feature_bin_tuple = (
            int(feature_bin_str.split("_")[0]),
            int(feature_bin_str.split("_")[1]),
        )
        report_dict[feature_bin_tuple] = value

    return report_dict


def load_individual_report(filepath: str) -> Dict:

    filepath = os.path.join(filepath, "report_individual_map.json")
    logg = GlobalLog("load_individual_report")

    assert os.path.exists(filepath), "{} does not exist".format(filepath)
    assert ".json" in filepath, "Filename {} is not a json file".format(filepath)
    logg.info("Loading {}".format(filepath))

    with open(filepath, "r") as f:
        report = json.loads(f.read())

    result = dict()
    population_json = report["population"]

    simulator_name = None
    for sim_name in SIMULATOR_NAMES:
        if sim_name in filepath:
            simulator_name = sim_name
            break
    assert (
        simulator_name is not None
    ), "Simulator name {} not found in filepath {}".format(SIMULATOR_NAMES, filepath)

    for features_bin_individuals in population_json:
        feature_bin = list(features_bin_individuals.keys())[0]
        result[feature_bin] = []
        for individual_dict in features_bin_individuals[feature_bin]:
            individual = _make_individual_from_report(
                filepath=filepath,
                individual_dict=individual_dict,
                simulator_name=simulator_name,
            )
            result[feature_bin].append(individual)

    return result


def write_individual_report(
    filepath: str,
    population: Dict,
) -> None:
    logg = GlobalLog("write_individual_report")
    report = {
        "population": [
            {
                "{}".format(feature_bin): [
                    individual.export() for individual in individuals
                ]
            }
            for feature_bin, individuals in population.items()
        ]
    }
    filename = os.path.join(filepath, "report_individual_map.json")
    if os.path.exists(filename):
        logg.info("Deleting existing {}".format(filename))
        os.remove(filename)

    with open(filename, "w") as f:
        json.dump(report, f, sort_keys=False, indent=4)


def write_mapelites_report(
    filepath: str,
    iterations: int,
    fitness_values: List[float],
    individuals: List[Individual],
    population: Dict,
) -> None:

    # all individuals in population by default
    ids_in_population = [individual.id for individual in individuals]
    if population is not None:
        ids_in_population = [individual.id for individual in population.values()]
    num_failures = sum([fitness_value < 0 for fitness_value in fitness_values])
    num_individuals_executed = len(individuals)

    report = {
        "iterations": iterations,
        "individuals_in_population": ids_in_population,
        "num_failures": num_failures,
        "num_individuals_executed": num_individuals_executed,
        "individuals": [individual.export() for individual in individuals],
    }

    filename = os.path.join(filepath, "report_iterations_{}.json".format(iterations))
    with open(filename, "w") as f:
        json.dump(report, f, sort_keys=False, indent=4)


def get_name_min_and_max_2d_features(
    individuals: List[Individual],
) -> Tuple[str, str, int, int, int, int]:
    assert (
        len(individuals) > 0
    ), "There should be at least an individual in the input list"

    feature_x_name = individuals[0].get_features()[0].name
    feature_y_name = individuals[0].get_features()[1].name

    # get max and min of the two features
    features_bin_x = [
        individual.get_features()[0].feature_bin for individual in individuals
    ]
    max_feature_x = max(features_bin_x)
    min_feature_x = min(features_bin_x)

    features_bin_y = [
        individual.get_features()[1].feature_bin for individual in individuals
    ]
    max_feature_y = max(features_bin_y)
    min_feature_y = min(features_bin_y)

    return (
        feature_x_name,
        feature_y_name,
        min_feature_x,
        max_feature_x,
        min_feature_y,
        max_feature_y,
    )


def get_individuals_and_reports_from_dirs(
    report_dirs: List[str],
) -> Tuple[List[Dict], List[Individual]]:
    individuals_in_population = []
    reports = []
    for report_dir in report_dirs:
        if "_all" in report_dir:
            individual_report = load_individual_report(filepath=report_dir)
            individuals = [
                individual
                for feature_bin in individual_report.keys()
                for individual in individual_report[feature_bin]
            ]
            ids = [individual.id for individual in individuals]

            for feature_bin in individual_report.keys():
                print("Feature bin: {}".format(feature_bin))
                for quality_metric in QUALITY_METRICS_NAMES:
                    behavioural_metrics_dicts = [
                        individual.get_behavioural_metrics()
                        for individual in individual_report[feature_bin]
                    ]
                    values = []
                    if len(behavioural_metrics_dicts) > 0:
                        behavioural_metrics_names = list(
                            behavioural_metrics_dicts[0].keys()
                        )
                        for behavioural_metric_name in behavioural_metrics_names:
                            if (
                                behavioural_metric_name in quality_metric
                                or quality_metric in behavioural_metric_name
                            ):
                                values = [
                                    make_quality_metric(
                                        quality_metric=quality_metric,
                                        behavioural_metric_name=behavioural_metric_name,
                                        behavioural_metrics=behavioural_metrics_dict,
                                    )
                                    for behavioural_metrics_dict in behavioural_metrics_dicts
                                ]
                    print(
                        "Quality metric: {}, values: {}".format(quality_metric, values)
                    )

            # turn individual_report in mapelites_report
            mapelites_report = dict()
            mapelites_report["ids_in_population"] = ids
            mapelites_report["individuals"] = individuals

            individuals_in_population.extend(individuals)
            reports.append(mapelites_report)
        else:
            all_report_files = glob.glob(
                os.path.join(report_dir, "report_iterations_*.json")
            )
            report_files = list(
                filter(
                    lambda rf: int(rf[rf.rindex("_") + 1 : rf.rindex(".")]) >= 0,
                    all_report_files,
                )
            )
            if len(report_files) == 1:
                report_file = report_files[0]
            elif len(report_files) == 2:
                report_file = report_files[1]
            else:
                raise RuntimeError(
                    "Number of report files {} not supported".format(len(report_files))
                )
            report = load_mapelites_report(filepath=report_file)
            reports.append(report)
            individuals_in_population.extend(
                list(
                    filter(
                        lambda ind: ind.id in report["ids_in_population"],
                        report["individuals"],
                    )
                )
            )
    return reports, individuals_in_population


def get_resized_maps_and_their_individuals(
    reports: List[Dict], individuals: List[Individual], quality_metric: str = None
) -> Tuple[List[Dict], List[Individual], Dict]:
    (
        feature_x_name,
        feature_y_name,
        min_feature_x,
        max_feature_x,
        min_feature_y,
        max_feature_y,
    ) = get_name_min_and_max_2d_features(individuals=individuals)

    resized_maps = []
    all_individuals_in_population = []
    fill_value = None
    for report in reports:
        individuals = list(
            filter(
                lambda ind: ind.id in report["ids_in_population"], report["individuals"]
            )
        )
        resized_map, resized_map_counts, fill_value = resize_map_of_elites(
            x_axis_min=min_feature_x,
            x_axis_max=max_feature_x,
            y_axis_min=min_feature_y,
            y_axis_max=max_feature_y,
            individuals=individuals,
            quality_metric=quality_metric,
        )
        resized_maps.append(resized_map)
        all_individuals_in_population.extend(individuals)

    all_feature_bins_with_value = [
        feature_bin
        for resized_map in resized_maps
        for feature_bin in resized_map.keys()
        if resized_map[feature_bin] != fill_value
    ]
    dict_feature_bins_with_values = dict(Counter(all_feature_bins_with_value))
    assert (
        len(all_feature_bins_with_value) > 0
    ), "There should be at least a feature bin with a value"

    return resized_maps, all_individuals_in_population, dict_feature_bins_with_values


def build_quality_metrics_map_from_report(
    filepath: str, quality_metric: str
) -> Tuple[Dict[Tuple[int, int], float], List[Individual], str, str]:
    logg = GlobalLog("build_quality_metrics_map_from_report")

    report = load_mapelites_report(filepath=filepath)

    individuals = list(
        filter(lambda ind: ind.id in report["ids_in_population"], report["individuals"])
    )

    (
        feature_x_name,
        feature_y_name,
        min_feature_x,
        max_feature_x,
        min_feature_y,
        max_feature_y,
    ) = get_name_min_and_max_2d_features(individuals=individuals)

    logg.info(
        "Feature x: {} in [{}, {}], Feature y: {} in [{}, {}]".format(
            feature_x_name,
            min_feature_x,
            max_feature_x,
            feature_y_name,
            min_feature_y,
            max_feature_y,
        )
    )

    # resized_map is the quality_metric_map since there is only one report and no other merging to do
    quality_metric_map, quality_metric_map_counts, fill_value = resize_map_of_elites(
        x_axis_min=min_feature_x,
        x_axis_max=max_feature_x,
        y_axis_min=min_feature_y,
        y_axis_max=max_feature_y,
        individuals=individuals,
        quality_metric=quality_metric,
    )

    return quality_metric_map, individuals, feature_x_name, feature_y_name


def build_quality_metrics_map_from_reports(
    report_name_suffix: str,
    filepath: str,
    env_name: str,
    str_datetime: str,
    num_runs: int,
    quality_metric: str,
) -> Tuple[Dict[Tuple[int, int], float], List[Individual], str, str]:
    logg = GlobalLog("build_quality_metrics_map_from_reports")

    report_dirs = glob.glob(
        os.path.join(
            filepath, "{}_{}_{}_*".format(report_name_suffix, env_name, str_datetime)
        )
    )
    # remove the all directories that summarizes all the runs if they exist
    # report_dirs = list(filter(lambda report_dir: "_all" not in report_dir, report_dirs))

    assert (
        len(report_dirs) == num_runs + 1
    ), "Number of report dirs {} != given number of runs {}".format(
        len(report_dirs), num_runs + 1
    )
    reports, individuals = get_individuals_and_reports_from_dirs(
        report_dirs=report_dirs
    )

    resized_maps, all_individuals, feature_bins_with_values = (
        get_resized_maps_and_their_individuals(
            reports=reports, individuals=individuals, quality_metric=quality_metric
        )
    )

    (
        feature_x_name,
        feature_y_name,
        min_feature_x,
        max_feature_x,
        min_feature_y,
        max_feature_y,
    ) = get_name_min_and_max_2d_features(individuals=all_individuals)

    logg.info(
        "Feature x: {} in [{}, {}], Feature y: {} in [{}, {}]".format(
            feature_x_name,
            min_feature_x,
            max_feature_x,
            feature_y_name,
            min_feature_y,
            max_feature_y,
        )
    )

    keys = resized_maps[0].keys()
    quality_metric_map = dict()
    population = dict()

    for key in keys:
        quality_metrics = [
            resized_map[key]
            for resized_map in resized_maps
            if resized_map[key] != FILL_VALUE
        ]
        if key in feature_bins_with_values:
            quality_metric_map[key] = np.mean(quality_metrics)
        elif len(quality_metrics) > 0:
            # if the key is not
            assert (
                len(quality_metrics) == 1
            ), "The number of valued keys != {} should be 1. Found: {}".format(
                FILL_VALUE, len(quality_metrics)
            )
            quality_metric_map[key] = quality_metrics[0]
        else:
            quality_metric_map[key] = FILL_VALUE

        individuals_with_key = list(
            filter(
                lambda individual: key
                == tuple(
                    [feature.get_value() for feature in individual.get_features()]
                ),
                all_individuals,
            )
        )
        population[key] = individuals_with_key

    quality_metric_map_filepath = os.path.join(
        filepath, "{}_{}_{}_all".format(report_name_suffix, env_name, str_datetime)
    )
    os.makedirs(name=quality_metric_map_filepath, exist_ok=True)

    # # cannot plot the quality metric map because we need the bounds (min and max value) for each metric in order to
    # # normalize the values for visualization. Such bounds will be known once all the maps from all the simulators are
    # # available.
    #
    # write_individual_report(
    #     filepath=quality_metric_map_filepath,
    #     population=population
    # )

    return quality_metric_map, all_individuals, feature_x_name, feature_y_name


def plot_and_save_probability_map(
    report_name_suffix: str,
    filepath: str,
    env_name: str,
    str_datetime: str,
    num_runs: int,
) -> None:

    logg = GlobalLog("plot_and_save_probability_map")

    logg.info("Plot map of elites end repetitions")

    report_dirs = glob.glob(
        os.path.join(
            filepath, "{}_{}_{}_*".format(report_name_suffix, env_name, str_datetime)
        )
    )
    # remove the all directory that summarizes all the runs if it exits
    report_dirs = list(filter(lambda report_dir: "_all" not in report_dir, report_dirs))

    assert (
        len(report_dirs) == num_runs
    ), "Number of report dirs {} > given number of runs {}".format(
        len(report_dirs), num_runs
    )

    reports, individuals = get_individuals_and_reports_from_dirs(
        report_dirs=report_dirs
    )

    resized_maps, all_individuals, feature_bins_with_values = (
        get_resized_maps_and_their_individuals(reports=reports, individuals=individuals)
    )

    (
        feature_x_name,
        feature_y_name,
        min_feature_x,
        max_feature_x,
        min_feature_y,
        max_feature_y,
    ) = get_name_min_and_max_2d_features(individuals=all_individuals)

    logg.info(
        "Feature x: {} in [{}, {}], Feature y: {} in [{}, {}]".format(
            feature_x_name,
            min_feature_x,
            max_feature_x,
            feature_y_name,
            min_feature_y,
            max_feature_y,
        )
    )

    keys = resized_maps[0].keys()
    probability_map = dict()
    population = dict()

    for key in keys:
        is_success_flags = [
            resized_map[key] > 0.0
            for resized_map in resized_maps
            if resized_map[key] != FILL_VALUE
        ]
        if key in feature_bins_with_values:
            probability_map[key] = np.mean(is_success_flags)
        elif len(is_success_flags) > 0:
            # if the key is not
            assert (
                len(is_success_flags) == 1
            ), "The number of valued keys != {} should be 1. Found: {}".format(
                FILL_VALUE, len(is_success_flags)
            )
            probability_map[key] = is_success_flags[0]
        else:
            probability_map[key] = FILL_VALUE

        individuals_with_key = list(
            filter(
                lambda individual: key
                == tuple(
                    [feature.get_value() for feature in individual.get_features()]
                ),
                all_individuals,
            )
        )
        population[key] = individuals_with_key

    probability_map_filepath = os.path.join(
        filepath, "{}_{}_{}_all".format(report_name_suffix, env_name, str_datetime)
    )
    os.makedirs(name=probability_map_filepath, exist_ok=True)

    plot_raw_map_of_elites(
        data=probability_map,
        filepath=probability_map_filepath,
        iterations=0,
        x_axis_label=feature_x_name,
        y_axis_label=feature_y_name,
        occupation_map=False,
    )

    plot_map_of_elites(
        data=probability_map,
        filepath=probability_map_filepath,
        iterations=0,
        x_axis_label=feature_x_name,
        y_axis_label=feature_y_name,
        min_value_cbar=0.0,
        max_value_cbar=1.0,
        occupation_map=False,
    )

    write_individual_report(filepath=probability_map_filepath, population=population)


def plot_probability_map(filepath: str, failure_probability: bool = False) -> None:
    report = load_individual_report(filepath=filepath)

    probability_map = dict()
    for feature_bin, individuals in report.items():
        fitness_values = [
            individual.get_fitness().get_value() for individual in individuals
        ]
        if failure_probability:
            flags = [fitness_value < 0.0 for fitness_value in fitness_values]
        else:
            flags = [fitness_value > 0.0 for fitness_value in fitness_values]
        if len(individuals) > 0:
            feature_x_name = individuals[0].get_features()[0].name
            feature_y_name = individuals[0].get_features()[1].name

        feature_bin_tuple = (
            int(feature_bin[feature_bin.find("(") + 1 : feature_bin.find(",")]),
            int(feature_bin[feature_bin.find(",") + 2 : feature_bin.find(")")]),
        )

        if len(fitness_values) > 0:
            probability_map[feature_bin_tuple] = np.mean(flags)
        else:
            probability_map[feature_bin_tuple] = FILL_VALUE

    plot_raw_map_of_elites(
        data=probability_map,
        filepath=filepath,
        iterations=0,
        x_axis_label=feature_x_name,
        y_axis_label=feature_y_name,
        occupation_map=False,
        failure_probability=failure_probability,
    )

    plot_map_of_elites(
        data=probability_map,
        filepath=filepath,
        iterations=0,
        x_axis_label=feature_x_name,
        y_axis_label=feature_y_name,
        min_value_cbar=0.0,
        max_value_cbar=1.0,
        occupation_map=False,
        failure_probability=failure_probability,
    )


def plot_raw_map_of_elites(
    data: Dict,
    filepath: str,
    iterations: int,
    x_axis_label: str,
    y_axis_label: str,
    occupation_map: bool = True,
    multiply_probabilities: bool = False,
    failure_probability: bool = False,
    quality_metric_merge: str = None,
    quality_metric: str = None,
    weighted_average_probabilities: bool = False,
) -> None:
    report_name = "raw_heatmap_occupation_{}_{}_iterations_{}.json".format(
        x_axis_label, y_axis_label, iterations
    )

    if not occupation_map:
        if multiply_probabilities:
            if failure_probability:
                report_name = "raw_heatmap_failure_probability_multiply_{}_{}_iterations_{}.json".format(
                    x_axis_label, y_axis_label, iterations
                )
            else:
                report_name = "raw_heatmap_success_probability_multiply_{}_{}_iterations_{}.json".format(
                    x_axis_label, y_axis_label, iterations
                )
        elif weighted_average_probabilities:
            if failure_probability:
                report_name = "raw_heatmap_failure_probability_weighted_{}_{}_iterations_{}.json".format(
                    x_axis_label, y_axis_label, iterations
                )
            else:
                report_name = "raw_heatmap_success_probability_weighted_{}_{}_iterations_{}.json".format(
                    x_axis_label, y_axis_label, iterations
                )
        else:
            if failure_probability:
                report_name = (
                    "raw_heatmap_failure_probability_{}_{}_iterations_{}.json".format(
                        x_axis_label, y_axis_label, iterations
                    )
                )
            else:
                report_name = (
                    "raw_heatmap_success_probability_{}_{}_iterations_{}.json".format(
                        x_axis_label, y_axis_label, iterations
                    )
                )

    if quality_metric is not None:
        if quality_metric_merge is not None:
            report_name = "raw_heatmap_{}_{}_{}_{}_iterations_{}.json".format(
                quality_metric,
                quality_metric_merge,
                x_axis_label,
                y_axis_label,
                iterations,
            )
        else:
            report_name = "raw_heatmap_{}_{}_{}_iterations_{}.json".format(
                quality_metric, x_axis_label, y_axis_label, iterations
            )

    filename = os.path.join(filepath, report_name)
    report = dict()
    for feature_bin, value in data.items():
        report["_".join(str(feature_int) for feature_int in feature_bin)] = value
    with open(filename, "w") as f:
        json.dump(report, f, sort_keys=False, indent=4)


def resize_map(
    x_axis_min: int,
    x_axis_max: int,
    y_axis_min: int,
    y_axis_max: int,
    report_dict: Dict,
    granularity: int = 1,
) -> Tuple[Dict, int]:

    if granularity == 1:
        x_axis_values = np.arange(start=x_axis_min, stop=x_axis_max + 1, dtype=np.int32)
        y_axis_values = np.arange(start=y_axis_min, stop=y_axis_max + 1, dtype=np.int32)
    else:
        raise NotImplementedError("Granularity values != 1 are not supported yet")

    fill_value = FILL_VALUE
    resized_map = dict()
    for x_axis_value in x_axis_values:
        for y_axis_value in y_axis_values:
            if (x_axis_value, y_axis_value) in report_dict:
                resized_map[(x_axis_value, y_axis_value)] = report_dict[
                    (x_axis_value, y_axis_value)
                ]
            else:
                resized_map[(x_axis_value, y_axis_value)] = fill_value

    return resized_map, fill_value


def get_values_from_individuals(
    individuals: List[Individual],
    quality_metric: str = None,
) -> List[float]:
    if quality_metric is not None:
        values = []
        behavioural_metrics_dicts = [
            individual.get_behavioural_metrics() for individual in individuals
        ]
        assert len(behavioural_metrics_dicts) > 0, "No behavioural metrics"
        behavioural_metrics_names = list(behavioural_metrics_dicts[0].keys())
        for behavioural_metric_name in behavioural_metrics_names:
            if (
                behavioural_metric_name in quality_metric
                or quality_metric in behavioural_metric_name
            ) and len(values) == 0:
                values = [
                    make_quality_metric(
                        quality_metric=quality_metric,
                        behavioural_metric_name=behavioural_metric_name,
                        behavioural_metrics=behavioural_metrics_dict,
                    )
                    for behavioural_metrics_dict in behavioural_metrics_dicts
                ]

        assert len(values) > 0, "Unknown quality metric: {}".format(quality_metric)
    else:
        values = [individual.get_fitness().get_value() for individual in individuals]

    return values


def resize_map_of_elites(
    x_axis_min: int,
    x_axis_max: int,
    y_axis_min: int,
    y_axis_max: int,
    individuals: List[Individual],
    granularity: int = 1,
    occupation_map: bool = False,
    failure_probability: bool = False,
    quality_metric: str = None,
) -> Tuple[Dict, Dict, int]:

    if granularity == 1:
        x_axis_values = np.arange(start=x_axis_min, stop=x_axis_max + 1, dtype=np.int32)
        y_axis_values = np.arange(start=y_axis_min, stop=y_axis_max + 1, dtype=np.int32)
    else:
        raise NotImplementedError("Granularity values != 1 are not supported yet")

    feature_bins = [
        tuple([feature.get_value() for feature in individual.get_features()])
        for individual in individuals
    ]
    values = get_values_from_individuals(
        individuals=individuals, quality_metric=quality_metric
    )

    resized_map = dict()
    resized_map_counts = dict()
    fill_value = FILL_VALUE

    for x_axis_value in x_axis_values:
        for y_axis_value in y_axis_values:
            if (x_axis_value, y_axis_value) in feature_bins:
                if occupation_map and quality_metric is None:
                    idx = feature_bins.index((x_axis_value, y_axis_value))
                    resized_map[(x_axis_value, y_axis_value)] = values[idx]
                    resized_map_counts[(x_axis_value, y_axis_value)] = 1
                else:
                    indices = [
                        i
                        for i, feature_bin in enumerate(feature_bins)
                        if (x_axis_value, y_axis_value) == feature_bin
                    ]
                    if quality_metric is not None:
                        quality_metric_values = [values[idx] for idx in indices]
                        resized_map[(x_axis_value, y_axis_value)] = np.mean(
                            quality_metric_values
                        )
                        resized_map_counts[(x_axis_value, y_axis_value)] = len(
                            quality_metric_values
                        )
                    else:
                        if failure_probability:
                            flags = [values[idx] < 0 for idx in indices]
                        else:
                            flags = [values[idx] > 0 for idx in indices]
                        resized_map[(x_axis_value, y_axis_value)] = np.mean(flags)
                        resized_map_counts[(x_axis_value, y_axis_value)] = len(flags)
            else:
                resized_map[(x_axis_value, y_axis_value)] = fill_value
                resized_map_counts[(x_axis_value, y_axis_value)] = 0

    return resized_map, resized_map_counts, fill_value


def plot_roc_curve(filepath: str, fpr: np.ndarray, tpr: np.ndarray) -> None:
    plt.figure()
    plt.plot(fpr, tpr, color="orange", label="ROC")
    plt.plot([0, 1], [0, 1], color="darkblue", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()

    plt.savefig(filepath, format="pdf")


def plot_map_of_elites(
    data: Dict,
    filepath: str,
    iterations: int,
    x_axis_label: str,
    y_axis_label: str,
    min_value_cbar: float,
    max_value_cbar: float,
    occupation_map: bool = True,
    multiply_probabilities: bool = False,
    failure_probability: bool = False,
    quality_metric: str = None,
    quality_metric_merge: str = None,
    weighted_average_probabilities: bool = False,
) -> None:
    plt.clf()
    plt.cla()

    ser = pd.Series(list(data.values()), index=pd.MultiIndex.from_tuples(data.keys()))
    # df = ser.unstack().fillna(0)
    df = ser.unstack().fillna(np.inf)

    # figure
    fig, ax = plt.subplots(figsize=(40, 40))

    colors = ["red", "gold", "green"]
    if not occupation_map:
        if failure_probability:
            label = "Failure probability"
            colors = ["green", "gold", "red"]
        else:
            label = "Success probability"
    else:
        label = "Fitness"

    if quality_metric is not None:
        colors = ["green", "gold", "red"]
        label = quality_metric

    cmap = LinearSegmentedColormap.from_list(name="test", colors=colors)

    # cmap = sns.cubehelix_palette(as_cmap=True)

    # Set the color for the under the limit to be white (0.0) so empty cells are not visualized
    # cmap.set_under('-1.0')
    # Plot NaN in white
    cmap.set_bad(color="white")

    # I had to transpose because the axes were swapped in the original implementation
    df = df.transpose()

    # sns.set(font_scale=7)

    ax = sns.heatmap(
        df,
        cmap=cmap,
        vmin=min_value_cbar,
        vmax=max_value_cbar,
        # cbar_kws={'label': label}
    )

    ax.invert_yaxis()
    ax.figure.axes[-1].set_ylabel(label, fontsize=80, weight="bold")
    ax.figure.axes[-1].set_yticklabels(
        ax.figure.axes[-1].get_ymajorticklabels(), fontsize=80, weight="bold"
    )
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=80, weight="bold")
    if y_axis_label == CURVATURE_FEATURE_NAME:
        ax.set_yticklabels(
            [
                str(round(int(label.get_text()) / 100, 2)) if i % 2 == 0 else ""
                for i, label in enumerate(ax.get_ymajorticklabels())
            ],
            fontsize=80,
            rotation=0,
            weight="bold",
        )
    plt.xlabel(x_axis_label, fontsize=80, weight="bold")
    plt.ylabel(y_axis_label, fontsize=80, weight="bold")

    # get figure to save to file
    if filepath:
        ht_figure = ax.get_figure()
        filename = "heatmap_occupation_{}_{}_iterations_{}".format(
            x_axis_label, y_axis_label, iterations
        )
        if not occupation_map:
            if multiply_probabilities:
                if failure_probability:
                    filename = "heatmap_failure_probability_multiply_{}_{}_iterations_{}".format(
                        x_axis_label, y_axis_label, iterations
                    )
                else:
                    filename = "heatmap_success_probability_multiply_{}_{}_iterations_{}".format(
                        x_axis_label, y_axis_label, iterations
                    )
            elif weighted_average_probabilities:
                if failure_probability:
                    filename = "heatmap_failure_probability_weighted_{}_{}_iterations_{}".format(
                        x_axis_label, y_axis_label, iterations
                    )
                else:
                    filename = "heatmap_success_probability_weighted_{}_{}_iterations_{}".format(
                        x_axis_label, y_axis_label, iterations
                    )
            else:
                if failure_probability:
                    filename = "heatmap_failure_probability_{}_{}_iterations_{}".format(
                        x_axis_label, y_axis_label, iterations
                    )
                else:
                    filename = "heatmap_success_probability_{}_{}_iterations_{}".format(
                        x_axis_label, y_axis_label, iterations
                    )

        if quality_metric is not None:
            if quality_metric_merge is not None:
                filename = "heatmap_{}_{}_{}_{}_iterations_{}".format(
                    quality_metric,
                    quality_metric_merge,
                    x_axis_label,
                    y_axis_label,
                    iterations,
                )
            else:
                filename = "heatmap_{}_{}_{}_iterations_{}".format(
                    quality_metric, x_axis_label, y_axis_label, iterations
                )

        fig_name = os.path.join(filepath, filename)
        ht_figure.savefig(fig_name)

    plt.clf()
    plt.cla()
    plt.close()


def save_images_of_individuals(
    filepath: str,
    population: Dict,
) -> None:
    ids_array = np.asarray(
        [
            individual.id
            for feature_bin, individuals in population.items()
            for individual in individuals
        ]
    )
    observations = []
    episode_lengths = []
    for individuals in population.values():
        for individual in individuals:
            episode_lengths.append(len(individual.get_observations()))
            observations.extend(individual.get_observations())

    numpy_dict = {
        "ids": ids_array,
        "episode_lengths": np.asarray(episode_lengths),
        "observations": np.asarray(observations),
    }

    np.savez(os.path.join(filepath, "observations.npz"), **numpy_dict)
