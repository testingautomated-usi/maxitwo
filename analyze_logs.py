import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point

from code_pipeline.visualization import RoadTestVisualizer
from config import SIMULATOR_NAMES
from global_log import GlobalLog
from self_driving.road_utils import get_road
from utils.dataset_utils import load_archive

parser = argparse.ArgumentParser()
parser.add_argument("--archive-path", help="Archive path", type=str, default="logs")
parser.add_argument(
    "--env-name",
    help="Simulator name",
    type=str,
    choices=SIMULATOR_NAMES,
    required=True,
)
parser.add_argument(
    "--output-dir-suffix", help="Simulator name", type=str, default=None
)
parser.add_argument(
    "--img-prefix", help="Name of the image files", type=str, default=None
)
parser.add_argument(
    "--archive-name", help="Archive name to analyze", type=str, required=True
)
parser.add_argument(
    "--plot-images", help="Plots images in folder", action="store_true", default=False
)
parser.add_argument("--num-images", help="Num images to plot", type=int, default=-1)
parser.add_argument(
    "--randomize",
    help="Does not plot the images in order",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--balance",
    help="The images that are going to be plotted have a balance between the steering angles",
    action="store_true",
    default=False,
)
args = parser.parse_args()

if __name__ == "__main__":

    assert os.path.exists(
        os.path.join(args.archive_path, args.archive_name)
    ), "Log {} does not exist".format(
        os.path.join(args.archive_path, args.archive_name)
    )
    assert ".npz" in args.archive_name, "Archive name must be a numpy archive"

    logg = GlobalLog("analyze_logs")

    output_dir = os.path.join(args.archive_path, args.env_name)
    if args.output_dir_suffix is not None:
        output_dir += "-" + args.output_dir_suffix

    os.makedirs(output_dir, exist_ok=True)

    numpy_dict = load_archive(
        archive_path=args.archive_path, archive_name=args.archive_name
    )
    # save observations and labels
    observations = numpy_dict["observations"]
    actions = numpy_dict["actions"]
    is_success_flags = []
    if "is_success_flags" in numpy_dict:
        is_success_flags = numpy_dict["is_success_flags"]
    episode_lengths = numpy_dict.get("episode_lengths", [])
    car_positions_x_episodes = numpy_dict.get("car_positions_x_episodes", [])
    car_positions_y_episodes = numpy_dict.get("car_positions_y_episodes", [])

    if args.plot_images and len(observations) > 0:
        logg.info("Plotting images")

        if args.randomize:
            logg.info("Randomize")
            np.random.shuffle(observations)

        bins = [(-1.0, -0.5), (-0.49, 0.0), (0.01, 0.5), (0.51, 1.0)]
        indices_images_bins = [[], [], [], []]

        for i in range(len(observations)):
            obs = observations[i]
            action = actions[i]

            if len(action.shape) > 1:
                action = action.squeeze()

            filename = "{}_".format(i)
            for j in range(len(action)):
                filename += str(float(action[j]))
                if j != len(action) - 1:
                    filename += "_"

            if not args.balance:
                if args.img_prefix is not None:
                    cv2.imwrite(
                        filename="{}.jpg".format(
                            os.path.join(output_dir, "{}{}".format(args.img_prefix, i))
                        ),
                        img=obs,
                    )
                else:
                    cv2.imwrite(
                        filename="{}.jpg".format(os.path.join(output_dir, filename)),
                        img=obs,
                    )
            else:
                if bins[0][0] <= action[0] <= bins[0][1]:
                    indices_images_bins[0].append(i)
                elif bins[1][0] <= action[0] <= bins[1][1]:
                    indices_images_bins[1].append(i)
                elif bins[2][0] <= action[0] <= bins[2][1]:
                    indices_images_bins[2].append(i)
                else:
                    indices_images_bins[3].append(i)

            if 0 < args.num_images - 1 == i:
                logg.info(
                    "Stop plotting, max num images reached {}".format(args.num_images)
                )
                break

        if args.balance:
            min_num_images_in_bins = np.min(
                [len(indices_images_bin) for indices_images_bin in indices_images_bins]
            )
            logg.info("Saving {} images for each bin".format(min_num_images_in_bins))
            total_images = 0
            for i, indices_images_bin in enumerate(indices_images_bins):
                for j in range(min_num_images_in_bins):

                    index_image_to_save = indices_images_bin[j]
                    obs = observations[index_image_to_save]

                    assert (
                        args.img_prefix is not None
                    ), "Specify the img_prefix parameter"
                    cv2.imwrite(
                        filename="{}.jpg".format(
                            os.path.join(
                                output_dir, "{}{}".format(args.img_prefix, total_images)
                            )
                        ),
                        img=obs,
                    )

                    if 0 < args.num_images - 1 == total_images:
                        logg.info(
                            "Stop plotting, max num images reached {}".format(
                                args.num_images
                            )
                        )
                        break

                    total_images += 1

    # FIXME: harmonize map_size
    road_test_visualizer = RoadTestVisualizer(map_size=250)
    tracks_concrete = []
    tracks_control_points = []
    if "tracks_concrete" in numpy_dict:
        tracks_concrete = numpy_dict["tracks_concrete"]
    if "tracks_control_points" in numpy_dict:
        tracks_control_points = numpy_dict["tracks_control_points"]

    for i in range(len(tracks_concrete)):
        track_concrete = tracks_concrete[i]
        road_points = [Point(item[0], item[1], item[2]) for item in track_concrete]
        road_width = track_concrete[0][-1]
        track_control_points = [
            Point(item[0], item[1], item[2]) for item in tracks_control_points[i]
        ]
        road = get_road(
            road_points=road_points,
            road_width=road_width,
            control_points=track_control_points,
            simulator_name=args.env_name,
        )

        if len(car_positions_x_episodes) > 0:
            car_trajectory_i = [
                (car_positions_x_episodes[i][j], car_positions_y_episodes[i][j])
                for j in range(len(car_positions_x_episodes[i]))
            ]
        else:
            car_trajectory_i = None

        road_test_visualizer.visualize_road_test(
            road=road,
            folder_path=output_dir,
            filename=(
                "road_{}_success_{}".format(i, is_success_flags[i])
                if len(is_success_flags) > 0
                else "road_{}".format(i)
            ),
            car_trajectory=car_trajectory_i,
        )

    if len(actions) > 0 and len(episode_lengths) > 0:
        logg.info("Visualizing steering angles of individual tracks")
        sum_episode_lengths = 0
        for i, episode_length in enumerate(episode_lengths):

            boxplot_dict = dict()
            boxplot_dict["steering_angles"] = actions[
                sum_episode_lengths : sum_episode_lengths + episode_length
            ][:, 0]

            plt.figure()
            plt.title(
                "{}-steering-angles-success-{}".format(
                    args.env_name, is_success_flags[i]
                )
            )
            plt.boxplot(x=boxplot_dict.values(), labels=boxplot_dict.keys())
            plt.ylim([-1, 1])
            plt.savefig(
                os.path.join(
                    output_dir,
                    "{}-steering-angle-distribution-success-{}.pdf".format(
                        i, is_success_flags[i]
                    ),
                ),
                format="pdf",
            )

            plt.close()

            sum_episode_lengths += episode_length

        plt.figure()
        plt.title(
            "distribution-all-steering-angles-sa-std_{:.4f}".format(
                np.mean(actions[:, 0])
            )
        )
        plt.hist(actions[:, 0])
        plt.xlim([-1, 1])
        plt.savefig(
            os.path.join(output_dir, "distribution-all-steering-angles.pdf"),
            format="pdf",
        )
        plt.close()
