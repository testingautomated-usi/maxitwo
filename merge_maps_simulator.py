import argparse
import os.path

import numpy as np

from config import SIMULATOR_NAMES
from global_log import GlobalLog
from test_generators.mapelites.config import QUALITY_METRICS_NAMES
from utils.report_utils import (
    get_values_from_individuals,
    load_individual_report,
    load_raw_map,
    plot_map_of_elites,
    plot_raw_map_of_elites,
    resize_map,
    write_individual_report,
)

parser = argparse.ArgumentParser()
parser.add_argument("--folder", help="Path of the folder where the logs are", type=str, default="logs")
parser.add_argument(
    "--env-name", help="Should be equal to the one of which we are merging the maps of", type=str, choices=SIMULATOR_NAMES
)
parser.add_argument("--filepaths", nargs="+", help="Paths of the folders where the reports are", type=str, required=True)
parser.add_argument("--output-dir", help="Output folder where the merged heatmap will be saved", type=str, default=None)
parser.add_argument(
    "--failure-probability",
    help="Whether to consider failure probability when building the map (default success_probability)",
    action="store_true",
    default=False,
)
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

args = parser.parse_args()

# FIXME: Merge multiple maps of the same simulator: consider refactoring with merge_mapelites.py
#  this file merges the raw maps directly using the average between the values,
#  since it merges maps of the same simulator
if __name__ == "__main__":

    logg = GlobalLog("merge_probability_maps")

    assert len(args.filepaths) == 2, "Number of probability maps to merge needs to be 2. Found: {}".format(len(args.filepaths))

    maps = []
    reports = []
    feature_x_name = None
    feature_y_name = None
    population_with_all_individuals = dict()

    for filepath in args.filepaths:
        complete_filepath = os.path.join(args.folder, filepath)
        map_report = load_individual_report(filepath=complete_filepath)
        reports.append(map_report)
        for feature_bin, individuals in map_report.items():
            if len(individuals) > 0:
                feature_x_name = individuals[0].get_features()[0].name
                feature_y_name = individuals[0].get_features()[1].name
            if feature_bin not in population_with_all_individuals:
                population_with_all_individuals[feature_bin] = []
            population_with_all_individuals[feature_bin].extend(individuals)

        maps.append(
            load_raw_map(
                filepath=complete_filepath,
                failure_probability=args.failure_probability,
                multiply_probabilities=False,
                quality_metric=args.quality_metric,
            )
        )

    all_feature_bins = [key for report in maps for key in report.keys()]
    x_axis_min = min([feature_bin[0] for feature_bin in all_feature_bins])
    x_axis_max = max([feature_bin[0] for feature_bin in all_feature_bins])
    y_axis_min = min([feature_bin[1] for feature_bin in all_feature_bins])
    y_axis_max = max([feature_bin[1] for feature_bin in all_feature_bins])

    resized_maps = []
    for i, single_map in enumerate(maps):
        resized_map, fill_value = resize_map(
            report_dict=single_map, x_axis_min=x_axis_min, x_axis_max=x_axis_max, y_axis_min=y_axis_min, y_axis_max=y_axis_max
        )
        resized_maps.append(resized_map)

    if args.output_dir is None:
        merged_heatmap_filepath = os.path.join(args.folder, "merged_{}".format("_".join(args.filepaths)))
    else:
        merged_heatmap_filepath = os.path.join(args.folder, args.output_dir)

    if os.path.exists(merged_heatmap_filepath):
        logg.warn("Directory {} already exists".format(merged_heatmap_filepath))

    os.makedirs(name=merged_heatmap_filepath, exist_ok=True)

    # assuming two maps
    assert len(resized_maps) == 2, "Only two maps are supported at the moment"

    resized_map_1 = resized_maps[0]
    resized_map_2 = resized_maps[1]

    valued_keys_1 = set(filter(lambda key: resized_map_1[key] != fill_value, resized_map_1.keys()))
    valued_keys_2 = set(filter(lambda key: resized_map_2[key] != fill_value, resized_map_2.keys()))

    bins_intersection = valued_keys_1.intersection(valued_keys_2)
    logg.info("# Keys that conflict: {}".format(len(bins_intersection)))

    values = dict()

    for k in resized_map_1.keys():
        value_1 = resized_map_1[k]
        value_2 = resized_map_2[k]
        if k in bins_intersection:
            assert str(k) in population_with_all_individuals, "Key {} not present in dictionary with all individuals"
            individuals = population_with_all_individuals[str(k)]

            values_individuals = get_values_from_individuals(individuals=individuals, quality_metric=args.quality_metric)
            if args.quality_metric is None:
                values_individuals_np = np.asarray(values_individuals)
                values_individuals = list(values_individuals_np < 0)

                values[k] = np.mean(values_individuals)

            else:
                assert args.min_quality_metric is not None, "min_quality_metric argument is needed for normalization"
                assert args.max_quality_metric is not None, "max_quality_metric argument is needed for normalization"
                assert (
                    args.min_quality_metric < args.max_quality_metric
                ), "Min quality metric {} > Max quality metric {}".format(args.min_quality_metric, args.max_quality_metric)

                normalized_values_individuals = []
                for value_individual in values_individuals:

                    normalized_value = (value_individual - args.min_quality_metric) / (
                        args.max_quality_metric - args.min_quality_metric
                    )

                    assert 0 <= normalized_value <= 1, "Value {}, original {}, not in bounds, key: {} bounds: ({}, {})".format(
                        normalized_value, value_individual, k, args.min_quality_metric, args.max_quality_metric
                    )

                    normalized_values_individuals.append(normalized_value)

                values[k] = np.mean(normalized_values_individuals)

            logg.info(
                "Conflict of key {} between the two maps: {} vs {}. Value in map: {}".format(
                    k, resized_map_1[k], resized_map_2[k], values[k]
                )
            )

        elif value_1 != fill_value:
            values[k] = value_1
        elif value_2 != fill_value:
            values[k] = value_2
        else:
            values[k] = fill_value

    logg.info("Saving maps and report in {}".format(merged_heatmap_filepath))

    plot_map_of_elites(
        data=values,
        filepath=merged_heatmap_filepath,
        iterations=0,
        x_axis_label=feature_x_name,
        y_axis_label=feature_y_name,
        min_value_cbar=0.0,
        max_value_cbar=1.0,
        occupation_map=False,
        failure_probability=args.failure_probability,
        quality_metric=args.quality_metric,
    )

    plot_raw_map_of_elites(
        data=values,
        filepath=merged_heatmap_filepath,
        iterations=0,
        x_axis_label=feature_x_name,
        y_axis_label=feature_y_name,
        occupation_map=False,
        failure_probability=args.failure_probability,
        quality_metric=args.quality_metric,
    )

    write_individual_report(filepath=merged_heatmap_filepath, population=population_with_all_individuals)
