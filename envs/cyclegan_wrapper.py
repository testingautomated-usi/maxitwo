import time
from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np
from PIL import Image

from config import SIMULATOR_NAMES
from cyclegan.data.base_dataset import get_transform
from cyclegan.models.test_model import TestModel
from cyclegan.util.util import tensor2im
from global_log import GlobalLog
from utils.dataset_utils import crop, image_size_after_crop


class CycleganWrapper(ABC):
    def __init__(self, env_name: str, cyclegan_model: TestModel, cyclegan_options: NamedTuple):
        assert env_name in SIMULATOR_NAMES, "Env name {} not in {}".format(env_name, SIMULATOR_NAMES)
        self.env_name = env_name
        self.cyclegan_model = cyclegan_model
        self.cyclegan_options = cyclegan_options
        self.logg = GlobalLog("cyclegan_wrapper")
        if cyclegan_options is not None:
            input_nc = cyclegan_options.output_nc if cyclegan_options.direction == "BtoA" else cyclegan_options.input_nc
            self.transform = get_transform(opt=cyclegan_options, grayscale=(input_nc == 1))

    @abstractmethod
    def stop_simulation(self) -> None:
        raise NotImplemented("Not implemented yet")

    @abstractmethod
    def restart_simulation(self) -> None:
        raise NotImplemented("Not implemented yet")

    def get_fake_image(self, obs: np.ndarray) -> np.ndarray:
        obs_cropped = crop(image=obs, env_name=self.env_name)
        obs_pil = Image.fromarray(obs_cropped)
        obs_tensor = self.transform(obs_pil)
        # unsqueeze(0) adds a leading dimension required for the image to pass through the cyclegan
        data = {"A": obs_tensor.unsqueeze(0), "A_paths": []}
        self.cyclegan_model.set_input(input=data)  # unpack data from data loader
        start_time = time.perf_counter()
        if len(self.cyclegan_options.gpu_ids) == 0:
            self.stop_simulation()
        self.cyclegan_model.test()  # run inference
        if len(self.cyclegan_model.gpu_ids) == 0:
            self.restart_simulation()
        self.logg.debug("Inference time: {:.2f}s".format(time.perf_counter() - start_time))

        for label, im_data in self.cyclegan_model.get_current_visuals().items():
            im = tensor2im(im_data)
            if label == "fake":
                assert (
                    self.cyclegan_options.original_image_size is not None
                    and len(self.cyclegan_options.original_image_size) > 0
                ), "Original image size not assigned"
                image_size = image_size_after_crop(
                    env_name=self.env_name, original_image_size=self.cyclegan_options.original_image_size
                )
                pil_image = Image.fromarray(im)
                # for some reason PIL wants the size parameter reversed
                pil_image = pil_image.resize(size=reversed(image_size))
                im = np.asarray(pil_image)
                return im
