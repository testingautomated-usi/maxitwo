from typing import List

from shapely.geometry import Point

from custom_types import Tuple4F
from self_driving.road import Road


class UdacityRoad(Road):

    def __init__(
        self, road_width: int, road_points: List[Point], control_points: List[Point]
    ):
        super().__init__(
            road_width=road_width,
            road_points=road_points,
            control_points=control_points,
        )

    def get_concrete_representation(self, to_plot: bool = False) -> List[Tuple4F]:
        if to_plot:
            return [
                (point.x, point.y, 1.90000000, self.road_width)
                for point in self.road_points
            ]
        return [
            (point.x, 1.90000000, point.y, self.road_width)
            for point in self.road_points
        ]

    def get_inverse_concrete_representation(
        self, to_plot: bool = False
    ) -> List[Tuple4F]:
        if to_plot:
            return [
                (point.x, point.y, 1.90000000, self.road_width)
                for point in reversed(self.road_points)
            ]
        return [
            (point.x, 1.90000000, point.y, self.road_width)
            for point in reversed(self.road_points)
        ]

    def serialize_concrete_representation(self, cr: List[Tuple4F]) -> str:
        # for Udacity we only need x, y, z
        to_string = []
        for t in cr:
            s = "{},{},{}".format(t[0], t[1], t[2])
            to_string.append(s)
        return "@".join(to_string)
