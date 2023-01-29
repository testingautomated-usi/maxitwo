import argparse
import os

import numpy as np

from config import BEAMNG_SIM_NAME, DONKEY_SIM_NAME, UDACITY_SIM_NAME
from global_log import GlobalLog
from test_generators.mapelites.config import QUALITY_METRICS_NAMES
from utils.report_utils import (
    FILL_VALUE,
    build_quality_metrics_map_from_report,
    build_quality_metrics_map_from_reports,
    get_values_from_individuals,
    plot_map_of_elites,
    plot_raw_map_of_elites,
)

parser = argparse.ArgumentParser()
parser.add_argument("--folder", help="Path of the folder where the logs are", type=str, default="logs")
parser.add_argument("--filepath", help="Path of the folders where the report is", type=str, required=True)
parser.add_argument(
    "--datetime-str-beamng", help="Datetime string used in the search folders of beamng", type=str, required=True
)
parser.add_argument(
    "--datetime-str-udacity", help="Datetime string used in the search folders of udacity", type=str, required=True
)
parser.add_argument(
    "--datetime-str-donkey", help="Datetime string used in the search folders of donkey", type=str, default=None
)

args = parser.parse_args()

if __name__ == "__main__":

    logg = GlobalLog("plot_quality_metrics_map")

    folder = args.folder
    report_name_suffix = "mapelites"

    str_datetime_env_name_num_runs_list = [
        ("{}".format(args.datetime_str_beamng), BEAMNG_SIM_NAME, 5),
        ("{}".format(args.datetime_str_udacity), UDACITY_SIM_NAME, 5),
    ]

    if args.datetime_str_donkey is not None:
        str_datetime_env_name_num_runs_list.append(("{}".format(args.datetime_str_donkey), DONKEY_SIM_NAME, 5))

    # FIXME: for simplicity at the moment the folders are hardcoded
    migration_folders_env_names = [
        ("{}/mapelites_migration_beamng_udacity_search".format(os.path.join(args.folder, args.filepath)), BEAMNG_SIM_NAME),
        ("{}/mapelites_migration_udacity_beamng_search".format(os.path.join(args.folder, args.filepath)), UDACITY_SIM_NAME),
        ("{}/mapelites_migration_donkey_beamng_search".format(os.path.join(args.folder, args.filepath)), DONKEY_SIM_NAME),
        ("{}/mapelites_migration_donkey_udacity_search".format(os.path.join(args.folder, args.filepath)), DONKEY_SIM_NAME),
    ]
    if args.datetime_str_donkey is not None:
        migration_folders_env_names.extend(
            [
                (
                    "{}/mapelites_migration_beamng_donkey_search".format(os.path.join(args.folder, args.filepath)),
                    BEAMNG_SIM_NAME,
                ),
                (
                    "{}/mapelites_migration_udacity_donkey_search".format(os.path.join(args.folder, args.filepath)),
                    UDACITY_SIM_NAME,
                ),
            ]
        )

    quality_metric_dicts_across_simulators = dict()
    for str_datetime, env_name, num_runs in str_datetime_env_name_num_runs_list:
        quality_metric_dict = dict()
        for quality_metric in QUALITY_METRICS_NAMES:
            quality_metric_map, individuals, feature_name_x, feature_name_y = build_quality_metrics_map_from_reports(
                report_name_suffix=report_name_suffix,
                filepath=folder,
                env_name=env_name,
                str_datetime=str_datetime,
                num_runs=num_runs,
                quality_metric=quality_metric,
            )
            values_individuals = get_values_from_individuals(individuals=individuals, quality_metric=quality_metric)
            min_quality_metric = min(values_individuals)
            max_quality_metric = max(values_individuals)
            logg.info(
                "Bounds {} for {}: ({}, {})".format(
                    quality_metric,
                    os.path.join(folder, "{}_{}_{}_all".format(report_name_suffix, env_name, str_datetime)),
                    min_quality_metric,
                    max_quality_metric,
                )
            )
            quality_metric_dict[quality_metric] = (
                quality_metric_map,
                feature_name_x,
                feature_name_y,
                min_quality_metric,
                max_quality_metric,
                os.path.join(folder, "{}_{}_{}_all".format(report_name_suffix, env_name, str_datetime)),
            )

        if env_name not in quality_metric_dicts_across_simulators:
            quality_metric_dicts_across_simulators[env_name] = []
        quality_metric_dicts_across_simulators[env_name].append(quality_metric_dict)

    for migration_folder, env_name in migration_folders_env_names:
        quality_metric_dict = dict()
        for quality_metric in QUALITY_METRICS_NAMES:
            report_file = os.path.join(migration_folder, "report_iterations_0.json")
            quality_metric_map, individuals, feature_name_x, feature_name_y = build_quality_metrics_map_from_report(
                filepath=report_file, quality_metric=quality_metric
            )
            # logg.info("Quality metric {}, map: {}".format(quality_metric, quality_metric_map))
            values_individuals = get_values_from_individuals(individuals=individuals, quality_metric=quality_metric)
            min_quality_metric = min(values_individuals)
            max_quality_metric = max(values_individuals)
            logg.info(
                "Bounds {} for {}: ({}, {})".format(
                    quality_metric, os.path.join(folder, migration_folder), min_quality_metric, max_quality_metric
                )
            )
            quality_metric_dict[quality_metric] = (
                quality_metric_map,
                feature_name_x,
                feature_name_y,
                min_quality_metric,
                max_quality_metric,
                migration_folder,
            )

        if env_name not in quality_metric_dicts_across_simulators:
            quality_metric_dicts_across_simulators[env_name] = []

        quality_metric_dicts_across_simulators[env_name].append(quality_metric_dict)

    quality_metrics_bounds = dict()
    for env_name, quality_metric_dicts in quality_metric_dicts_across_simulators.items():
        if args.datetime_str_donkey is not None:
            # cross-validation
            assert len(quality_metric_dicts) == 3, "{} has more or less than 3 folders. Found: {}".format(
                env_name, len(quality_metric_dicts)
            )
        else:
            assert len(quality_metric_dicts) == 2, "{} has more or less than 2 folders. Found: {}".format(
                env_name, len(quality_metric_dicts)
            )

        for quality_metric_dict in quality_metric_dicts:
            for quality_metric in quality_metric_dict.keys():
                if quality_metric not in quality_metrics_bounds:
                    # min and max initialization
                    quality_metrics_bounds[quality_metric] = (np.inf, -np.inf)

                current_minimum = quality_metrics_bounds[quality_metric][0]
                current_maximum = quality_metrics_bounds[quality_metric][1]
                quality_metric_map = quality_metric_dict[quality_metric][0]
                min_map = quality_metric_dict[quality_metric][3]
                max_map = quality_metric_dict[quality_metric][4]
                quality_metrics_bounds[quality_metric] = (
                    min(current_minimum, min_map),
                    max(current_maximum, max_map),
                )

    logg.info("Bounds for the quality metrics: {}".format(quality_metrics_bounds))

    # normalizing quality_metrics_maps
    for env_name, quality_metric_dicts in quality_metric_dicts_across_simulators.items():
        for quality_metric_dict in quality_metric_dicts:
            for quality_metric in quality_metric_dict.keys():
                quality_metric_map = quality_metric_dict[quality_metric][0]
                _max = quality_metrics_bounds[quality_metric][1]
                _min = quality_metrics_bounds[quality_metric][0]
                for key in quality_metric_map.keys():
                    if quality_metric_map[key] != FILL_VALUE:
                        quality_metric_map[key] = (quality_metric_map[key] - _min) / (_max - _min)
                        assert (
                            0 <= quality_metric_map[key] <= 1
                        ), "Problem in normalization: value != [0, 1]: {}, ({}, {})".format(
                            quality_metric_map[key], _min, _max
                        )

    for env_name, quality_metric_dicts in quality_metric_dicts_across_simulators.items():
        for quality_metric_dict in quality_metric_dicts:
            for quality_metric in quality_metric_dict.keys():
                quality_metric_map = quality_metric_dict[quality_metric][0]
                feature_name_x = quality_metric_dict[quality_metric][1]
                feature_name_y = quality_metric_dict[quality_metric][2]
                folder_for_saving_map = quality_metric_dict[quality_metric][5]
                logg.info("Saving map for quality metric {} in {}".format(quality_metric, folder_for_saving_map))
                plot_raw_map_of_elites(
                    data=quality_metric_map,
                    filepath=folder_for_saving_map,
                    iterations=0,
                    x_axis_label=feature_name_x,
                    y_axis_label=feature_name_y,
                    quality_metric=quality_metric,
                )

                plot_map_of_elites(
                    data=quality_metric_map,
                    filepath=folder_for_saving_map,
                    iterations=0,
                    x_axis_label=feature_name_x,
                    y_axis_label=feature_name_y,
                    min_value_cbar=0.0,
                    max_value_cbar=1.0,
                    quality_metric=quality_metric,
                )
