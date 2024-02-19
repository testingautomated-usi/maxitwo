import logging

from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera


from envs.beamng.config import BASE_PORT
from envs.beamng.decal_road import DecalRoad
from envs.beamng.simulation_data import SimulationParams
from global_log import GlobalLog
from self_driving.pose import Pose
from self_driving.road_points import List4DTuple, RoadPoints


class BeamNGCamera:
    def __init__(self, beamng: BeamNGpy, name: str, camera: Camera = None):
        self.name = name
        self.pose: Pose = Pose()
        self.camera = camera
        if not self.camera:
            self.camera = Camera(
                (0, 0, 0),
                (0, 0, 0),
                120,
                (1280, 1280),
                colour=True,
                depth=True,
                annotation=True,
            )
        self.beamng = beamng

    def get_rgb_image(self):
        self.camera.pos = self.pose.pos
        self.camera.direction = self.pose.rot
        cam = self.beamng.render_cameras()
        img = cam[self.name]["colour"].convert("RGB")
        return img


class BeamNGBrewer:
    def __init__(
        self,
        beamng_home=None,
        beamng_user=None,
        reuse_beamng=True,
        road_nodes: List4DTuple = None,
        add_to_port: int = 0,
        autopilot: bool = False,
    ):
        self.reuse_beamng = reuse_beamng
        if self.reuse_beamng:
            # This represents the running BeamNG simulator. Since we use launch=True this should automatically
            # shut down when the main python process exits or when we call self.beamng_process.stop()
            self.beamng_process = BeamNGpy(
                "localhost", BASE_PORT + add_to_port, home=beamng_home, user=beamng_user
            )
            self.beamng_process = self.beamng_process.open(launch=True)

        # This is used to bring up each simulation without restarting the simulator
        self.beamng = BeamNGpy(
            "localhost", 64256 + add_to_port, home=beamng_home, user=beamng_user
        )

        self.beamng_home = beamng_home
        self.beamng_user = beamng_user

        self.road_nodes = None
        self.decal_road = None
        self.road_points = None

        self.scenario = None

        self.logg = GlobalLog("BeamNGBrewer")

        # We need to wait until this point otherwise the BeamNG loggers's level will be (re)configured by BeamNGpy
        self.logg.info("Disabling BEAMNG logs")
        for id in [
            "beamngpy",
            "beamngpy.beamngpycommon",
            "beamngpy.BeamNGpy",
            "beamngpy.beamng",
            "beamngpy.Scenario",
            "beamngpy.Vehicle",
            "beamngpy.Camera",
        ]:
            logger = logging.getLogger(id)
            logger.setLevel(logging.CRITICAL)
            logger.disabled = True

        self.vehicle: Vehicle = None
        self.camera: BeamNGCamera = None
        if road_nodes:
            self.setup_road_nodes(road_nodes)

        if autopilot:
            # in order to collect the same number of frames as in the other simulators
            steps = 20
        else:
            steps = 60  # real time

        self.params = SimulationParams(
            beamng_steps=steps, delay_msec=int(steps * 0.05 * 1000)
        )
        self.vehicle_start_pose = Pose()

    def setup_road_nodes(self, road_nodes):
        self.road_nodes = road_nodes
        self.decal_road: DecalRoad = DecalRoad("street_1").add_4d_points(road_nodes)
        self.road_points = RoadPoints().add_middle_nodes(road_nodes)

    def setup_vehicle(self) -> Vehicle:
        assert self.vehicle is None
        self.vehicle = Vehicle("ego_vehicle", model="etkc", licence="TIG", color="Red")
        return self.vehicle

    def setup_scenario_camera(self) -> BeamNGCamera:
        assert self.camera is None
        self.camera = BeamNGCamera(self.beamng, "brewer_camera")
        return self.camera

    # TODO COnsider to transform brewer into a ContextManager or get rid of it...
    def bring_up(self):

        if self.reuse_beamng:
            # This assumes BeamNG is already running
            self.beamng.open(launch=False)
        else:
            self.beamng.open(launch=True)

        # After 1.18 to make a scenario one needs a running instance of BeamNG
        self.scenario = Scenario("tig", "tigscenario")
        if self.vehicle:
            self.scenario.add_vehicle(
                self.vehicle,
                pos=self.vehicle_start_pose.pos,
                rot=self.vehicle_start_pose.rot,
            )

        if self.camera:
            self.scenario.add_camera(self.camera.camera, self.camera.name)

        self.scenario.make(self.beamng)

        self.beamng.set_deterministic()

        self.beamng.load_scenario(self.scenario)

        self.beamng.start_scenario()

        # Pause the simulator only after loading and starting the scenario
        self.beamng.pause()

    def __del__(self):
        if self.beamng:
            try:
                self.beamng.close()
            except:
                pass
