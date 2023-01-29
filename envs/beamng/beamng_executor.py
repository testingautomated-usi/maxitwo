import os.path
import time
import traceback
from typing import Dict, Tuple

import numpy as np
from shapely.geometry import Point

from code_pipeline.visualization import RoadTestVisualizer
from custom_types import ObserveData
from envs.beamng.beamng_brewer import BeamNGBrewer

# maps is a global variable in the module, which is initialized to Maps()
from envs.beamng.beamng_car_cameras import BeamNGCarCameras
from envs.beamng.beamng_tig_maps import LevelsFolder, maps
from envs.beamng.beamng_waypoint import BeamNGWaypoint
from envs.beamng.config import BEAMNG_VERSION, MAX_SPEED_BEAMNG, MIN_SPEED_BEAMNG
from envs.beamng.decal_road import DecalRoad
from envs.beamng.simulation_data import SimulationDataRecord
from envs.beamng.simulation_data_collector import SimulationDataCollector
from envs.beamng.vehicle_state_reader import VehicleStateReader
from global_log import GlobalLog
from test_generators.mapelites.individual import Individual
from test_generators.test_generator import TestGenerator

FloatDTuple = Tuple[float, float, float, float]


class BeamngExecutor:
    def __init__(
        self,
        test_generator: TestGenerator = None,
        oob_tolerance: float = 0.95,
        max_speed: int = MAX_SPEED_BEAMNG,
        beamng_home: str = None,
        beamng_user: str = None,
        road_visualizer: RoadTestVisualizer = None,
        add_to_port: int = 0,
        logs_folder_name: str = None,
        autopilot: bool = False,
    ):

        self.logger = GlobalLog("BeamngExecutor")
        self.risk_value = 0.7
        self.oob_tolerance = oob_tolerance
        self.maxspeed = max_speed

        self.vehicle = None

        self.track = None

        self.brewer: BeamNGBrewer = None
        self.beamng_home = beamng_home
        self.beamng_user = beamng_user
        self.test_generator = test_generator

        self.autopilot = autopilot

        self.add_to_port = add_to_port
        self.logs_folder_name = logs_folder_name
        self.speed_limit = max_speed
        self.max_speed_in_ms = max_speed * 0.277778
        self.max_speed = max_speed
        self.waypoints_crossed_index = 0
        self.waypoints = []

        assert os.path.exists(self.beamng_home), "Path to beamng home does not exist: {}".format(beamng_home)
        assert os.path.exists(self.beamng_user), "Path to beamng user does not exist: {}".format(beamng_user)

        # TODO Add checks with default setup. This requires a call to BeamNGpy resolve  (~/Documents/BeamNG.research)
        if self.beamng_user is not None and not os.path.exists(os.path.join(self.beamng_user, "tech.key")):
            self.logger.warn(
                "{} is missing but is required to use BeamNG.research".format(os.path.join(self.beamng_user, "tech.key"))
            )

        # Runtime Monitor about relative movement of the car
        self.last_observation = None
        # Not sure how to set this... How far can a car move in 250 ms at 5Km/h
        self.min_delta_position = 1.0
        self.road_visualizer = road_visualizer

        self.vehicle_state_reader = None
        self.sim_data_collector = None
        self.waypoint_goal = None
        self.last_throttle = 0.0
        self.agent_state = None
        self.original_image = None
        self.episode_steps = 0

    @staticmethod
    def get_node_coords(node):
        return node[0], node[1], node[2]

    def points_distance(self, p1, p2):
        return np.linalg.norm(np.subtract(self.get_node_coords(p1), self.get_node_coords(p2)))

    def _is_the_car_moving(self, last_state):
        """Check if the car moved in the past 10 seconds"""

        # Has the position changed
        if self.last_observation is None:
            self.last_observation = last_state
            return True

        # If the car moved since the last observation, we store the last state and move one
        if (
            Point(self.last_observation.pos[0], self.last_observation.pos[1]).distance(
                Point(last_state.pos[0], last_state.pos[1])
            )
            > self.min_delta_position
        ):
            self.last_observation = last_state
            return True
        else:
            # How much time has passed since the last observation?
            if last_state.timer - self.last_observation.timer > 10.0:
                return False
            else:
                return True

    def reset(self, skip_generation: bool = False, individual: Individual = None) -> None:

        self.agent_state = None
        self.last_throttle = 0.0
        self.last_observation = None
        self.original_image = None
        self.episode_steps = 0
        self.waypoints_crossed_index = 0
        self.waypoints.clear()

        if self.brewer is None:
            self.brewer = BeamNGBrewer(
                beamng_home=self.beamng_home,
                beamng_user=self.beamng_user,
                add_to_port=self.add_to_port,
                autopilot=self.autopilot,
            )
            self.vehicle = self.brewer.setup_vehicle()
        else:
            # FIXME
            self.end_iteration()

        if not skip_generation:

            if individual is None:
                assert self.test_generator is not None, "Test generator is not instantiated"

                start_time = time.perf_counter()
                self.logger.debug("Start generating track")
                self.track = self.test_generator.generate()
                self.logger.debug("Track generated: {:.2f}s".format(time.perf_counter() - start_time))
            else:
                self.track = individual.get_representation()

            nodes = self.track.get_concrete_representation()
            # For the execution we need the interpolated points
            self.brewer.setup_road_nodes(nodes)
            self.waypoints = [self.get_node_coords(node) for node in nodes]
            self.waypoint_goal = BeamNGWaypoint("waypoint_goal", self.get_node_coords(nodes[-1]))

            # Note This changed since BeamNG.research
            beamng_levels = LevelsFolder(os.path.join(self.beamng_user, BEAMNG_VERSION, "levels"))
            maps.beamng_levels = beamng_levels
            maps.beamng_map = maps.beamng_levels.get_map("tig")
            # maps.print_paths()

            maps.install_map_if_needed()
            maps.beamng_map.generated().write_items(self.brewer.decal_road.to_json() + "\n" + self.waypoint_goal.to_json())

            cameras = BeamNGCarCameras()
            self.vehicle_state_reader = VehicleStateReader(
                self.vehicle, self.brewer.beamng, additional_sensors=cameras.cameras_array
            )

            self.brewer.vehicle_start_pose = self.brewer.road_points.vehicle_start_pose()

            if self.logs_folder_name is not None:
                simulation_id = time.strftime("%Y-%m-%d--%H-%M-%S", time.localtime())
                name = "{}/{}".format(self.logs_folder_name, simulation_id)
            else:
                name = None

            self.sim_data_collector = SimulationDataCollector(
                self.vehicle,
                self.brewer.beamng,
                self.brewer.decal_road,
                self.brewer.params,
                vehicle_state_reader=self.vehicle_state_reader,
                simulation_name=name,
            )

            # TODO: Hacky - Not sure what's the best way to set this...
            self.sim_data_collector.oob_monitor.tolerance = self.oob_tolerance

            self.sim_data_collector.get_simulation_data().start()

            self.brewer.bring_up()

            if self.autopilot:
                script = self.calculate_script(self.brewer.road_points.middle)
                # Trick: we start from the road center
                self.vehicle.ai_set_script(script[4:])

    def take_action(self, steering: float, throttle: float = None) -> None:
        """
        * ``steering``: Rotation of the steering wheel, from -1.0 to 1.0.
         * ``throttle``: Intensity of the throttle, from 0.0 to 1.0.
        """
        try:
            if not self.autopilot:

                if throttle is None:
                    if self.sim_data_collector is None:
                        speed = 0
                    else:
                        speed = self.sim_data_collector.states[-1].vel_kmh

                    if speed > self.speed_limit:
                        self.speed_limit = MIN_SPEED_BEAMNG  # slow down
                    else:
                        self.speed_limit = self.max_speed

                    throttle = np.clip(a=1.0 - steering**2 - (speed / self.speed_limit) ** 2, a_min=0.0, a_max=1.0)

                self.last_throttle = throttle

                self.vehicle.control(throttle=float(throttle), steering=float(steering), brake=0)

            self.brewer.beamng.step(self.brewer.params.beamng_steps)

        except ConnectionAbortedError:
            self.brewer.beamng.resume()

    def is_game_over(self) -> Tuple[bool, int]:
        self.sim_data_collector.collect_current_data(oob_bb=True)
        last_state: SimulationDataRecord = self.sim_data_collector.states[-1]
        # Target point reached
        target_reached = self.points_distance(last_state.pos, self.waypoint_goal.position) < 8.0
        is_oob = last_state.is_oob
        if target_reached or is_oob:
            return True, 1 if target_reached else 0
        return False, 0

    def observe(self) -> ObserveData:
        try:
            self.episode_steps += 1
            done, success_bit = self.is_game_over()
            img = self.vehicle_state_reader.sensors["cam_center"]["colour"].convert("RGB")
            self.original_image = np.array(img)
            info = {
                "is_success": success_bit,
                "track": self.track,
                "pos": (self.sim_data_collector.states[-1].pos[0], self.sim_data_collector.states[-1].pos[1]),
                "speed": self.sim_data_collector.states[-1].vel_kmh,
                "lateral_position": self.sim_data_collector.oob_monitor.oob_distance(),
            }

            if self.autopilot:
                info["steering"] = self.sim_data_collector.states[-1].steering_input
                info["throttle"] = self.sim_data_collector.states[-1].throttle_input
            else:
                info["throttle"] = self.last_throttle

            return np.array(img), done, info
        except ConnectionAbortedError:
            self.brewer.beamng.resume()

    def send_agent_state(self, agent_state: Dict):
        self.agent_state = agent_state

    def end_iteration(self) -> None:
        try:
            self.sim_data_collector.save(save_road_image=False)
            self.sim_data_collector.take_car_picture_if_needed()
        except Exception as ex:
            pass

        try:
            if self.brewer:
                self.brewer.beamng.stop_scenario()
        except Exception as ex:
            traceback.print_exception(type(ex), ex, ex.__traceback__)

    def close(self) -> None:
        if self.brewer:
            try:
                self.brewer.beamng.close()
            except Exception as ex:
                traceback.print_exception(type(ex), ex, ex.__traceback__)
            self.brewer = None

    #### autopilot functions ####

    # x is -y and *angle direction is reversed*
    @staticmethod
    def get_rotation(road: DecalRoad):
        v1 = road.nodes[0][:2]
        v2 = road.nodes[1][:2]
        v = np.subtract(v1, v2)
        deg = np.degrees(np.arctan2([v[0]], [v[1]]))
        return 0, 0, deg

    @staticmethod
    def get_script_point(p1, p2) -> Tuple[Tuple, Tuple]:
        a = np.subtract(p2[0:2], p1[0:2])

        # calculate the vector which length is half the road width
        v = (a / np.linalg.norm(a)) * p1[3] / 4

        # add normal vectors
        r = p1[0:2] + np.array([v[1], -v[0]])
        return tuple(r)

    # Calculate the points to guide the AI from the road points
    def calculate_script(self, road_points):
        script_points = [self.get_script_point(road_points[i], road_points[i + 1]) for i in range(len(road_points) - 1)]
        assert len(script_points) == len(road_points) - 1
        # Get the last script point
        script_points += [self.get_script_point(road_points[-1], road_points[-2])]
        assert len(script_points) == len(road_points)
        orig = script_points[0]

        script = [{"x": orig[0], "y": orig[1], "z": 0.5, "t": 0}]
        i = 1
        # time = 0.18
        time = 0.18
        # goal = len(street_1.nodes) - 1
        # goal = len(brewer.road_points.right) - 1
        goal = len(script_points) - 1

        while i < goal:
            node = {
                # 'x': street_1.nodes[i][0],
                # 'y': street_1.nodes[i][1],
                # 'x': brewer.road_points.right[i][0],
                # 'y': brewer.road_points.right[i][1],
                "x": script_points[i][0],
                "y": script_points[i][1],
                "z": 0.5,
                "t": time,
            }
            script.append(node)
            i += 1
            time += 0.18
        return script

    def distance(self, p1, p2):
        return np.linalg.norm(np.subtract(self.get_node_coords(p1), self.get_node_coords(p2)))
