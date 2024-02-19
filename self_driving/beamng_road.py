from typing import List

from shapely.geometry import Point

from custom_types import Tuple4F
from self_driving.road import Road


class BeamNGRoad(Road):

    def __init__(
        self, road_width: int, road_points: List[Point], control_points: List[Point]
    ):
        super().__init__(
            road_width=road_width,
            road_points=road_points,
            control_points=control_points,
        )

    def get_concrete_representation(self, to_plot: bool = False) -> List[Tuple4F]:
        return [
            (point.x, point.y, -28.0, self.road_width) for point in self.road_points
        ]

    def get_inverse_concrete_representation(
        self, to_plot: bool = False
    ) -> List[Tuple4F]:
        return [
            (point.x, point.y, -28.0, self.road_width)
            for point in reversed(self.road_points)
        ]

    def serialize_concrete_representation(self, cr: List[Tuple4F]) -> str:
        raise NotImplementedError("This method should not be needed for BEAMNG")
