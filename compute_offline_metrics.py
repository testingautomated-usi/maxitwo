import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance, wilcoxon
from statsmodels.stats.power import TTestIndPower

from effect_size import cohend
from global_log import GlobalLog
from utils.dataset_utils import load_archive

parser = argparse.ArgumentParser()
parser.add_argument("--archive-path", help="Path of the folder where the logs are", type=str, default="logs")
parser.add_argument("--archive-names", nargs="+", help="Paths of the folders where the reports are", type=str, required=True)
parser.add_argument("--output-dir", help="Output folder where the merged heatmap will be saved", type=str, default=None)
parser.add_argument(
    "--predict-throttle",
    help="Predict steering and throttle. Model to load must have been trained using an output dimension of 2",
    action="store_true",
    default=False,
)
parser.add_argument("--show-plot", help="Show histograms", action="store_true", default=False)

args = parser.parse_args()


def get_bin_positions(bins_edges: List[float]) -> List[float]:
    bins_positions = []
    for i in range(len(bins_edges) - 1):
        if len(bins_positions) == 0:
            bins_positions.append(bins_edges[i + 1] - bins_edges[i])
        else:
            bins_positions.append(bins_positions[-1] + bins_edges[i + 1] - bins_edges[i])
    return bins_positions


if __name__ == "__main__":

    assert len(args.archive_names) <= 3, "Cannot compute offline metrics among more than 3 maps"

    for archive_name in args.archive_names:
        assert os.path.exists(os.path.join(args.archive_path, archive_name)), "{} does not exist".format(
            os.path.join(args.archive_path, archive_name)
        )

    logg = GlobalLog("compute_offline_metrics")

    all_prediction_errors = []

    for i, archive_name in enumerate(args.archive_names):
        numpy_dict = load_archive(archive_path=args.archive_path, archive_name=archive_name)
        ith_predictions_errors = numpy_dict["prediction_errors"]
        concatenated_ith_prediction_errors = []
        if type(ith_predictions_errors[0]) == np.ndarray:
            for prediction_errors in ith_predictions_errors:
                if args.predict_throttle:
                    prediction_errors_list = list(prediction_errors.flatten())
                    concatenated_ith_prediction_errors.extend(prediction_errors_list)
                else:
                    concatenated_ith_prediction_errors.extend([prediction_error for prediction_error in prediction_errors])
        elif type(ith_predictions_errors[0]) == np.float64 or type(ith_predictions_errors[0]) == np.float32:
            concatenated_ith_prediction_errors.extend(ith_predictions_errors)
        else:
            raise RuntimeError("Unknown type of prediction error: {}".format(type(ith_predictions_errors[0])))

        all_prediction_errors.append(concatenated_ith_prediction_errors)

    prediction_errors_1 = all_prediction_errors[0]
    prediction_errors_2 = all_prediction_errors[1]

    range_prediction_error = (0.0, 0.6)
    step_size = 0.04
    bins = int(range_prediction_error[1] / step_size)

    plt.figure()
    plt.rcParams.update({"font.size": 35, "font.weight": "bold"})
    # it will be the env name: e.g. archive name "offline-evaluation-fake-beamng.npz"
    name_1 = args.archive_names[0].split("-")[-1].split(".")[0]
    name_2 = args.archive_names[1].split("-")[-1].split(".")[0]

    # https://stackoverflow.com/questions/24391892/printing-subscript-in-python
    if name_1 == "beamng":
        name_1 = "DS1".translate(str.maketrans("1", "₁"))
    elif name_1 == "udacity":
        name_1 = "DS2".translate(str.maketrans("2", "₂"))

    if name_2 == "donkey":
        name_2 = "HDT"

    n_1, bin_edges_1, patches_1 = plt.hist(
        x=prediction_errors_1, bins=bins, range=range_prediction_error, density=True, alpha=0.3, color="red", label=name_1
    )
    n_2, bin_edges_2, patches_2 = plt.hist(
        x=prediction_errors_2, bins=bins, range=range_prediction_error, density=True, alpha=0.3, color="blue", label=name_2
    )
    plt.legend()
    plt.xlabel("Error magnitude", weight="bold")
    plt.ylabel("Error percentage", weight="bold")

    plt.xlim(range_prediction_error)
    # plt.ylim([min(min(n_1), min(n_2)), max(max(n_1), max(n_2))])
    plt.ylim([0, 18])
    if len(args.archive_names) != 3 and args.show_plot:
        plt.show()
    else:
        plt.close()

    analysis = TTestIndPower()
    desired_power = 0.8
    alpha = 0.05

    if len(args.archive_names) == 3:
        prediction_errors_3 = all_prediction_errors[2]

        logg.info(
            "Providing more than 2 archives: merging the first two and computing metrics "
            "between the merged and the third archive"
        )

        prediction_errors_merged = prediction_errors_1 + prediction_errors_2

        name_3 = args.archive_names[2].split("-")[-1].split(".")[0]
        if name_3 == "donkey":
            name_3 = "HDT"

        plt.figure()
        plt.rcParams.update({"font.size": 35, "font.weight": "bold"})
        n_merged, bin_edges_merged, patches_merged = plt.hist(
            x=prediction_errors_merged,
            bins=bins,
            range=range_prediction_error,
            density=True,
            alpha=0.3,
            color="red",
            label="DSS",
        )
        n_3, bin_edges_3, patches_3 = plt.hist(
            x=prediction_errors_3, bins=bins, range=range_prediction_error, density=True, alpha=0.3, color="blue", label=name_3
        )
        plt.legend()
        plt.xlabel("Error magnitude", weight="bold")
        plt.ylabel("Error percentage", weight="bold")
        plt.xlim(range_prediction_error)
        # plt.ylim([min(min(n_1), min(n_2)), max(max(n_1), max(n_2))])
        plt.ylim([0, 18])
        if args.show_plot:
            plt.show()
        else:
            plt.close()

        bin_positions_merged = get_bin_positions(bins_edges=bin_edges_merged)
        bin_positions_3 = get_bin_positions(bins_edges=bin_edges_3)

        distance = wasserstein_distance(
            u_values=bin_positions_merged, v_values=bin_positions_3, u_weights=n_merged, v_weights=n_3
        )
        logg.info("Distance: {}".format(distance))

        statistic, p_value = wilcoxon(x=n_3, y=n_merged)
        logg.info("Wilcoxon p-value: {}".format(p_value))

        estimate, magnitude = cohend(a=prediction_errors_merged, b=prediction_errors_3)
        logg.info("Cohen's d effect size among prediction errors: {}, {}".format(estimate, magnitude))

        if p_value > 0.05 and abs(estimate) > 0.0:
            power = analysis.power(
                effect_size=estimate, nobs1=len(prediction_errors_merged) + len(prediction_errors_3), alpha=alpha
            )
            logg.info("Parametric power at alpha {}: {}".format(alpha, power))

            nobs = analysis.solve_power(effect_size=estimate, power=desired_power, alpha=alpha)
            logg.info(
                "Number of observations required to have power {} at alpha {}: {}. "
                "Current number of observations: {}".format(
                    desired_power, alpha, nobs, len(prediction_errors_merged) + len(prediction_errors_3)
                )
            )

    else:

        bin_positions_1 = get_bin_positions(bins_edges=bin_edges_1)
        bin_positions_2 = get_bin_positions(bins_edges=bin_edges_2)

        distance = wasserstein_distance(u_values=bin_positions_1, v_values=bin_positions_2, u_weights=n_1, v_weights=n_2)
        logg.info("Distance: {}".format(distance))

        statistic, p_value = wilcoxon(x=n_1, y=n_2)
        logg.info("Wilcoxon p-value: {}".format(p_value))

        estimate, magnitude = cohend(a=prediction_errors_1, b=prediction_errors_2)
        logg.info("Cohen's d effect size among prediction errors: {}, {}".format(estimate, magnitude))

        if p_value > 0.05 and abs(estimate) > 0.0:

            power = analysis.power(
                effect_size=estimate, nobs1=len(prediction_errors_1) + len(prediction_errors_2), alpha=alpha
            )
            logg.info("Parametric power at alpha {}: {}".format(alpha, power))

            nobs = analysis.solve_power(effect_size=estimate, power=desired_power, alpha=alpha)
            logg.info(
                "Number of observations required to have power {} at alpha {}: {}. "
                "Current number of observations: {}".format(
                    desired_power, alpha, nobs, len(prediction_errors_1) + len(prediction_errors_2)
                )
            )
