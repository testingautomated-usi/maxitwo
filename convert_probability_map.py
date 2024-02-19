import argparse
import os.path

from utils.report_utils import plot_probability_map

parser = argparse.ArgumentParser()
parser.add_argument(
    "--folder", help="Path of the folder where the logs are", type=str, default="logs"
)
parser.add_argument(
    "--filepath",
    help="Path of the folders where the report is",
    type=str,
    required=True,
)
parser.add_argument(
    "--failure-probability",
    help="Whether to consider failure probability when building the map (default success_probability)",
    action="store_true",
    default=False,
)

args = parser.parse_args()

if __name__ == "__main__":

    filepath = args.filepath
    filepath = os.path.join(args.folder, filepath)

    plot_probability_map(
        filepath=filepath, failure_probability=args.failure_probability
    )
