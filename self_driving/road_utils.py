from typing import List

from shapely.geometry import Point

from config import BEAMNG_SIM_NAME, DONKEY_SIM_NAME, MOCK_SIM_NAME, SIMULATOR_NAMES, UDACITY_SIM_NAME
from self_driving.beamng_road import BeamNGRoad
from self_driving.catmull_rom import catmull_rom
from self_driving.donkey_road import DonkeyRoad
from self_driving.road import Road
from self_driving.udacity_road import UdacityRoad


def get_road(road_points: List[Point], control_points: List[Point], road_width: int, simulator_name: str) -> Road:
    assert simulator_name in SIMULATOR_NAMES, "Simulator name {} not supported. Choose between: {}".format(
        simulator_name, SIMULATOR_NAMES
    )

    if simulator_name == BEAMNG_SIM_NAME:
        return BeamNGRoad(road_width=road_width, road_points=road_points, control_points=control_points)
    if simulator_name == DONKEY_SIM_NAME:
        return DonkeyRoad(road_width=road_width, road_points=road_points, control_points=control_points)
    if simulator_name == UDACITY_SIM_NAME:
        return UdacityRoad(road_width=road_width, road_points=road_points, control_points=control_points)

    if simulator_name == MOCK_SIM_NAME:
        # just for simplicity returning DonkeyRoad; for now I do not need a mock road
        return DonkeyRoad(road_width=road_width, road_points=road_points, control_points=control_points)

    raise RuntimeError("Unknown simulator name: {}".format(simulator_name))


def get_road_from_control_points(
    control_points: List[Point], road_width: int, num_sampled_points: int, simulator_name: str
) -> Road:
    road_points = catmull_rom(
        [(cp.x, cp.y, cp.z, road_width) for cp in control_points[1:]], num_spline_points=num_sampled_points
    )
    road_points = [Point(rp[0], rp[1]) for rp in road_points]

    return get_road(
        road_points=road_points, control_points=control_points, road_width=road_width, simulator_name=simulator_name
    )
