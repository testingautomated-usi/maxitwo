import datetime
import json
import os
import shutil
import uuid
from collections import namedtuple
from time import sleep
from typing import List, Union, Tuple
from pathlib import Path

import cv2
import numpy as np

from envs.beamng.decal_road import DecalRoad
from self_driving.road_imagery import RoadImagery

SimulationDataRecordProperties = [
    "timer",
    "pos",
    "dir",
    "vel",
    "steering",
    "steering_input",
    "brake",
    "brake_input",
    "throttle",
    "throttle_input",
    "wheelspeed",
    "vel_kmh",
    "is_oob",
    "oob_counter",
    "max_oob_percentage",
    "oob_distance",
    "oob_percentage",
]

SimulationDataRecord = namedtuple(
    "SimulationDataRecord", SimulationDataRecordProperties
)
SimulationDataRecords = List[SimulationDataRecord]

SimulationParams = namedtuple("SimulationParameters", ["beamng_steps", "delay_msec"])


def delete_folder_recursively(path: Union[str, Path]):
    path = str(path)
    if not os.path.exists(path):
        return
    assert os.path.isdir(path), path
    print(f"Removing [{path}]")
    shutil.rmtree(path, ignore_errors=True)

    # sometimes rmtree fails to remove files
    for tries in range(20):
        if os.path.exists(path):
            sleep(0.1)
            shutil.rmtree(path, ignore_errors=True)

    if os.path.exists(path):
        shutil.rmtree(path)

    if os.path.exists(path):
        raise Exception(f"Unable to remove folder [{path}]")


class SimulationInfo:
    start_time: str
    end_time: str
    success: bool
    exception_str: str
    computer_name: str
    ip_address: str
    id: str


class SimulationData:
    f_info = "info"
    f_params = "params"
    f_road = "road"
    f_records = "records"

    def __init__(self, simulation_name: str):
        self.name = simulation_name
        if simulation_name is not None:
            root: Path = Path(
                os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            )
            self.simulations: Path = root.joinpath("simulations")
            self.path_root: Path = self.simulations.joinpath(simulation_name)
            self.path_json: Path = self.path_root.joinpath("simulation.full.json")
            self.path_partial: Path = self.path_root.joinpath("simulation.partial.tsv")
            self.path_road_img: Path = self.path_root.joinpath("road")

        self.id: str = None
        self.params: SimulationParams = None
        self.road: DecalRoad = None
        self.states: SimulationDataRecord = None
        self.info: SimulationInfo = None
        self.exception_str = None
        self.images: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    @property
    def n(self):
        return len(self.states)

    def set(
        self,
        params: SimulationParams,
        road: DecalRoad,
        states: SimulationDataRecords,
        info: SimulationInfo = None,
    ):
        self.params = params
        self.road = road
        if info:
            self.info = info
        else:
            self.info = SimulationInfo()
            self.info.id = str(uuid.uuid4())
        self.states = states

    def clean(self):
        delete_folder_recursively(self.path_root)

    def save(self, save_road_image: bool = True):

        if self.name is not None:
            self.path_root.mkdir(parents=True, exist_ok=True)
            with open(self.path_json, "w") as f:
                f.write(
                    json.dumps(
                        {
                            self.f_params: self.params._asdict(),
                            self.f_info: self.info.__dict__,
                            self.f_road: self.road.to_dict(),
                            self.f_records: [r._asdict() for r in self.states],
                        }
                    )
                )

            with open(self.path_partial, "w") as f:
                sep = "\t"
                f.write(sep.join(SimulationDataRecordProperties) + "\n")
                gen = (r._asdict() for r in self.states)
                gen2 = (
                    sep.join([str(d[key]) for key in SimulationDataRecordProperties])
                    + "\n"
                    for d in gen
                )
                f.writelines(gen2)

            if save_road_image:
                road_imagery = RoadImagery.from_sample_nodes(self.road.nodes)
                road_imagery.save(self.path_road_img.with_suffix(".jpg"))
                road_imagery.save(self.path_road_img.with_suffix(".svg"))

            if len(self.images) > 0:
                os.makedirs(name=self.path_root.__str__() + "/images", exist_ok=True)
                for i in range(len(self.images)):
                    img_center = self.images[i][0]
                    img_left = self.images[i][1]
                    img_right = self.images[i][2]
                    img_center = cv2.cvtColor(img_center, cv2.COLOR_BGR2RGB)
                    img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
                    img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(
                        filename=self.path_root.__str__()
                        + "/images/{}_center.jpg".format(i),
                        img=img_center,
                    )
                    cv2.imwrite(
                        filename=self.path_root.__str__()
                        + "/images/{}_left.jpg".format(i),
                        img=img_left,
                    )
                    cv2.imwrite(
                        filename=self.path_root.__str__()
                        + "/images/{}_right.jpg".format(i),
                        img=img_right,
                    )

    def load(self) -> "SimulationData":
        with open(self.path_json, "r") as f:
            obj = json.loads(f.read())
        info = SimulationInfo()

        info.__dict__ = obj.get(self.f_info, {})
        self.set(
            SimulationParams(**obj[self.f_params]),
            DecalRoad.from_dict(obj[self.f_road]),
            [SimulationDataRecord(**r) for r in obj[self.f_records]],
            info=info,
        )
        return self

    def load_from_json(self, path_json):
        with open(path_json, "r") as f:
            obj = json.loads(f.read())
        info = SimulationInfo()

        info.__dict__ = obj.get(self.f_info, {})
        self.set(
            SimulationParams(**obj[self.f_params]),
            DecalRoad.from_dict(obj[self.f_road]),
            [SimulationDataRecord(**r) for r in obj[self.f_records]],
            info=info,
        )
        return self

        pass

    def complete(self) -> bool:
        return self.path_json.exists()

    def min_oob_distance(self) -> float:
        return min(state.oob_distance for state in self.states)

    def start(self):
        self.info.success = None
        self.info.start_time = str(datetime.datetime.now())
        try:
            import platform

            self.info.computer_name = platform.node()
        except Exception as ex:
            self.info.computer_name = str(ex)

    def end(self, success: bool, exception=None):
        self.info.end_time = str(datetime.datetime.now())
        self.info.success = success
        if exception:
            self.exception_str = str(exception)
