import os
from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import LineString, Polygon
from shapely.affinity import translate, rotate
from descartes import PolygonPatch
from math import atan2, pi, degrees


# https://stackoverflow.com/questions/34764535/why-cant-matplotlib-plot-in-a-different-thread
from self_driving.road import Road


class RoadTestVisualizer:
    """
        Visualize and Plot RoadTests
    """

    little_triangle = Polygon([(10, 0), (0, -5), (0, 5), (10, 0)])
    square = Polygon([(5, 5), (5, -5), (-5, -5), (-5, 5), (5,5)])

    def __init__(self, map_size):
        self.map_size = map_size
        self.last_submitted_test_figure = None

        # Make sure there's a windows and does not block anything when calling show
        plt.ion()
        plt.show()

    def _setup_figure(self):
        if self.last_submitted_test_figure is not None:
            # Make sure we operate on the right figure
            plt.figure(self.last_submitted_test_figure.number)
            plt.clf()
        else:
            self.last_submitted_test_figure = plt.figure()

        # plt.gcf().set_title("Last Generated Test")
        plt.gca().set_aspect('equal', 'box')
        plt.gca().set(xlim=(-30, self.map_size + 30), ylim=(-30, self.map_size + 30))

    def visualize_road_test(
            self,
            road: Road,
            folder_path: str = os.getcwd(),
            filename: str = 'road_0',
            invert: bool = False,
            car_trajectory: List[Tuple[float]] = None,
            plot_control_points: bool = False
    ) -> None:

        self._setup_figure()

        plt.draw()
        plt.pause(0.001)
        
        # Plot the map. Trying to re-use an artist in more than one Axes which is supported
        map_patch = patches.Rectangle((0, 0), self.map_size, self.map_size, linewidth=1, edgecolor='black', facecolor='none')
        plt.gca().add_patch(map_patch)

        # Road Geometry.
        if not invert:
            road_poly = LineString([(t[0], t[1]) for t in road.get_concrete_representation(to_plot=True)]).buffer(8.0, cap_style=2, join_style=2)
        else:
            road_poly = LineString([(t[0], t[1]) for t in road.get_inverse_concrete_representation(to_plot=True)]).buffer(8.0, cap_style=2, join_style=2)

        if car_trajectory is not None or plot_control_points:
            # blur the road such that the trajectory of the car is visible on the road
            road_patch = PolygonPatch(road_poly, fc='gray', ec='dimgray', alpha=0.4)  # ec='#555555', alpha=0.5, zorder=4)
        else:
            road_patch = PolygonPatch(road_poly, fc='gray', ec='dimgray')  # ec='#555555', alpha=0.5, zorder=4)

        plt.gca().add_patch(road_patch)

        # Interpolated Points
        if not invert:
            sx = [t[0] for t in road.get_concrete_representation(to_plot=True)]
            sy = [t[1] for t in road.get_concrete_representation(to_plot=True)]
        else:
            sx = [t[0] for t in road.get_inverse_concrete_representation(to_plot=True)]
            sy = [t[1] for t in road.get_inverse_concrete_representation(to_plot=True)]

        plt.plot(sx, sy, 'yellow')

        # Plot the little triangle indicating the starting position of the ego-vehicle
        delta_x = sx[1] - sx[0]
        delta_y = sy[1] - sy[0]

        current_angle = atan2(delta_y, delta_x)

        rotation_angle = degrees(current_angle)
        transformed_fov = rotate(self.little_triangle, origin=(0, 0), angle=rotation_angle)
        transformed_fov = translate(transformed_fov, xoff=sx[0], yoff=sy[0])
        plt.plot(*transformed_fov.exterior.xy, color='black')

        # Plot the little square indicating the ending position of the ego-vehicle
        delta_x = sx[-1] - sx[-2]
        delta_y = sy[-1] - sy[-2]

        current_angle = atan2(delta_y, delta_x)

        rotation_angle = degrees(current_angle)
        transformed_fov = rotate(self.square, origin=(0, 0), angle=rotation_angle)
        transformed_fov = translate(transformed_fov, xoff=sx[-1], yoff=sy[-1])
        plt.plot(*transformed_fov.exterior.xy, color='black')

        plt.draw()

        if car_trajectory is not None:
            car_trajectory_x = [cr_item[0] for cr_item in car_trajectory]
            car_trajectory_y = [cr_item[1] for cr_item in car_trajectory]
            plt.scatter(car_trajectory_x, car_trajectory_y, color="red")

        if plot_control_points:
            control_points_xs = [cp.x for cp in road.control_points]
            control_points_ys = [cp.y for cp in road.control_points]
            print(control_points_xs)
            print(control_points_ys)
            plt.scatter(control_points_xs, control_points_ys, color="red", marker="*", s=50)

        plt.pause(0.001)
        plt.savefig(os.path.join(folder_path, '{}.png'.format(filename)))




