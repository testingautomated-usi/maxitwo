import argparse
import glob
import os
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from global_log import GlobalLog
from test_generators.mapelites.config import QUALITY_METRICS_NAMES
from test_generators.mapelites.individual import Individual
from utils.report_utils import (
    get_name_min_and_max_2d_features,
    load_mapelites_report,
    load_raw_map,
    plot_roc_curve,
    resize_map,
    resize_map_of_elites,
)

parser = argparse.ArgumentParser()
parser.add_argument("--folder", help="Path of the folder where the logs are", type=str, default="logs")
parser.add_argument("--filepaths", nargs="+", help="Paths of the folders where the reports are", type=str, required=True)
parser.add_argument("--output-dir", help="Output folder where the merged heatmap will be saved", type=str, default=None)
parser.add_argument("--load-probability-map", help="Load probability map", action="store_true", default=False)
parser.add_argument(
    "--multiply-probabilities",
    help="Multiply probabilities when merging the probability maps (by default the probabilities are averaged)",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--failure-probability",
    help="Whether to consider failure probability when building the map (default success_probability)",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--only-failures",
    help="Whether to consider only failures when computing the correlation between two maps",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--quality-metric", help="Name of the quality metric", type=str, choices=QUALITY_METRICS_NAMES, default=None
)
parser.add_argument(
    "--quality-metric-merge",
    help="How to merge the quality metric maps",
    type=str,
    choices=["avg", "min", "max"],
    default=None,
)
parser.add_argument(
    "--failure-predicts-quality-metric",
    help='Whether failure probability predicts quality metric. If "no" then quality metric predicts failure probability',
    type=str,
    choices=["yes", "no"],
    default="yes",
)

args = parser.parse_args()

if __name__ == "__main__":

    assert len(args.filepaths) <= 2, "Cannot merge more than 2 maps"

    for filepath in args.filepaths:
        assert os.path.exists(os.path.join(args.folder, filepath)), "{} does not exist".format(
            os.path.join(args.folder, filepath)
        )

    logg = GlobalLog("compute_mapelites_correlation")

    reports = []

    for i, filepath in enumerate(args.filepaths):
        if args.load_probability_map or args.quality_metric is not None:
            report_filepath = os.path.join(args.folder, filepath)
            assert os.path.exists(report_filepath), "Report file {} does not exist".format(report_filepath)
            quality_metric = None
            if args.quality_metric is not None:
                if (args.failure_predicts_quality_metric == "yes" and i == 1) or (
                    args.failure_predicts_quality_metric == "no" and i == 0
                ):
                    quality_metric = args.quality_metric

            raw_map = load_raw_map(
                filepath=report_filepath,
                failure_probability=args.failure_probability,
                multiply_probabilities=args.multiply_probabilities,
                quality_metric=quality_metric,
                quality_metric_merge=args.quality_metric_merge,
            )
            reports.append(raw_map)
        else:
            all_report_files = glob.glob(os.path.join(args.folder, filepath, "report_iterations_*.json"))
            report_files = list(filter(lambda rf: int(rf[rf.rindex("_") + 1 : rf.rindex(".")]) >= 0, all_report_files))
            if len(report_files) == 1:
                report_file = report_files[0]
            elif len(report_files) == 2:
                report_file = report_files[1]
            else:
                raise RuntimeError("Number of report files {} not supported".format(len(report_files)))
            reports.append(load_mapelites_report(filepath=report_file))

    numpy_maps = []

    if args.load_probability_map or args.quality_metric is not None:

        all_feature_bins = [key for report in reports for key in report.keys()]
        x_axis_min = min([feature_bin[0] for feature_bin in all_feature_bins])
        x_axis_max = max([feature_bin[0] for feature_bin in all_feature_bins])
        y_axis_min = min([feature_bin[1] for feature_bin in all_feature_bins])
        y_axis_max = max([feature_bin[1] for feature_bin in all_feature_bins])

        resized_maps = []
        for i, report in enumerate(reports):
            resized_map, fill_value = resize_map(
                report_dict=report,
                x_axis_min=x_axis_min,
                x_axis_max=x_axis_max,
                y_axis_min=y_axis_min,
                y_axis_max=y_axis_max,
            )

            ser = pd.Series(list(resized_map.values()), index=pd.MultiIndex.from_tuples(resized_map.keys()))
            df = ser.unstack()
            numpy_map = df.to_numpy()
            numpy_maps.append(numpy_map)
    else:
        all_individuals_in_population: List[Individual] = []
        for i, report in enumerate(reports):
            all_individuals_in_population.extend(
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
        ) = get_name_min_and_max_2d_features(individuals=all_individuals_in_population)

        logg.info(
            "Feature x: {} in [{}, {}], Feature y: {} in [{}, {}]".format(
                feature_x_name, min_feature_x, max_feature_x, feature_y_name, min_feature_y, max_feature_y
            )
        )
        fill_value = None

        for i, report in enumerate(reports):

            if args.load_probability_map:
                individuals = [individual for feature_bin in report.keys() for individual in report[feature_bin]]
            else:
                individuals = list(filter(lambda ind: ind.id in report["ids_in_population"], report["individuals"]))

            resized_map, resized_map_counts, fill_value = resize_map_of_elites(
                x_axis_min=min_feature_x,
                x_axis_max=max_feature_x,
                y_axis_min=min_feature_y,
                y_axis_max=max_feature_y,
                individuals=individuals,
            )

            ser = pd.Series(list(resized_map.values()), index=pd.MultiIndex.from_tuples(resized_map.keys()))
            df = ser.unstack().fillna(fill_value)
            resized_map_numpy = df.to_numpy()
            numpy_maps.append(resized_map_numpy)

    # assuming there are two maps
    m1 = numpy_maps[0]
    m2 = numpy_maps[1]
    assert m1.shape == m2.shape, "The two maps do not have the same shape: {} != {}".format(m1.shape, m2.shape)
    flatten_m1 = m1.flatten()
    flatten_m2 = m2.flatten()

    # -2 should not be present neither in the occupation map (fitness min lateral position goes from -0.1 to 2.0)
    # nor in the probability map ([0.0, 1.0]). If the fitness changes then change also this value
    replace_fill_value_with_value = -2

    elements_equal = np.where(flatten_m1 == replace_fill_value_with_value)[0]
    assert (
        len(elements_equal) == 0
    ), "Find another value rather than {} to replace {} for correlation computation. -2 is present in the map: {}".format(
        replace_fill_value_with_value, fill_value, flatten_m1
    )

    elements_equal = np.where(flatten_m2 == replace_fill_value_with_value)[0]
    assert (
        len(elements_equal) == 0
    ), "Find another value rather than {} to replace {} for correlation computation. -2 is present in the map: {}".format(
        replace_fill_value_with_value, fill_value, flatten_m2
    )

    flatten_m1 = np.where(flatten_m1 == fill_value, replace_fill_value_with_value, flatten_m1)
    flatten_m2 = np.where(flatten_m2 == fill_value, replace_fill_value_with_value, flatten_m2)

    if args.load_probability_map or args.quality_metric is not None:
        indices_m1_with_fill_value = np.where(flatten_m1 == replace_fill_value_with_value)[0]
        indices_m2_with_fill_value = np.where(flatten_m2 == replace_fill_value_with_value)[0]
        assert np.alltrue(
            np.equal(indices_m1_with_fill_value, indices_m2_with_fill_value)
        ), "The two maps fill value positions must be equal. {} != {}".format(
            indices_m1_with_fill_value, indices_m2_with_fill_value
        )

    filtered_m1, filtered_m2 = [], []
    for i, val in enumerate(flatten_m1):
        if val != replace_fill_value_with_value and flatten_m2[i] != replace_fill_value_with_value:
            if args.only_failures:
                if (args.failure_probability and val > 0.0 and flatten_m2[i] > 0.0) or (
                    not args.failure_probability and val < 1.0 and flatten_m2[i] < 1.0
                ):
                    filtered_m1.append(val)
                    filtered_m2.append(flatten_m2[i])
            else:
                filtered_m1.append(val)
                filtered_m2.append(flatten_m2[i])

    filtered_m1 = np.asarray(filtered_m1)
    filtered_m2 = np.asarray(filtered_m2)

    logg.info("Num values in map: {}. Num values in filtered map: {}".format(len(flatten_m1), len(filtered_m1)))

    corr, p_value = pearsonr(x=filtered_m1, y=filtered_m2)
    logg.info("Correlation filtered: {}, p-value {}".format(corr, p_value))

    ground_truth = filtered_m2 > 0
    auc = roc_auc_score(y_true=ground_truth, y_score=filtered_m1)
    logg.info("AUC ROC: {}".format(auc))

    fpr, tpr, thresholds = roc_curve(y_true=ground_truth, y_score=filtered_m1)
    auc_roc_filename = "auc_roc"
    filepath1 = args.filepaths[0].replace("mapelites_", "")
    filepath2 = args.filepaths[1].replace("mapelites_", "")
    if args.quality_metric is not None:
        auc_roc_filename += "_{}".format(args.quality_metric)
        if args.quality_metric_merge is not None:
            auc_roc_filename += "_{}".format(args.quality_metric_merge)
    else:
        auc_roc_filename += "_failure"
        if args.multiply_probabilities:
            auc_roc_filename += "_multiply"

    auc_roc_curve_filepath = os.path.join(args.folder, "{}_{}_{}.pdf".format(auc_roc_filename, filepath1, filepath2))
    logg.info("Saving ROC curve in {}".format(auc_roc_curve_filepath))
    plot_roc_curve(filepath=auc_roc_curve_filepath, fpr=fpr, tpr=tpr)

    average_precision = average_precision_score(y_true=ground_truth, y_score=filtered_m1)
    logg.info("Average precision: {}".format(average_precision))
    precision, recall, thresholds = precision_recall_curve(y_true=ground_truth, probas_pred=filtered_m1)
    logg.debug(
        "Precision-recall baseline: {} ({}/{})".format(
            np.sum(ground_truth) / len(ground_truth), np.sum(ground_truth), len(ground_truth)
        )
    )
