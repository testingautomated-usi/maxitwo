import os
import time

import numpy as np

from config import BEAMNG_SIM_NAME
from cyclegan.data import create_dataset
from cyclegan.data.base_dataset import get_transform
from cyclegan.models import create_model
from cyclegan.util.util import get_base_and_test_default_options, tensor2im

from PIL import Image

from utils.dataset_utils import image_size_after_crop

if __name__ == "__main__":

    cyclegan_experiment_name = "beamng"
    gpu_ids = "0"
    cyclegan_checkpoints_dir = f"cyclegan{os.sep}checkpoints"
    cyclegan_epoch = "35"

    opt = get_base_and_test_default_options(
        name=cyclegan_experiment_name,
        gpu_ids=gpu_ids,
        checkpoints_dir=cyclegan_checkpoints_dir,
        epoch=cyclegan_epoch,
        dataroot="cyclegan",
        dataset_mode="archive",
        archive_filepath=f"logs{os.sep}beamng-2022_05_31_18_50_03-archive-agent-autopilot-seed-0-episodes-1.npz",
        max_dataset_size=float("inf"),
    )

    dataset = create_dataset(
        opt
    )  # create a dataset given opt.dataset_mode and other options
    cyclegan_model = create_model(
        opt
    )  # create a model given opt.model and other options
    cyclegan_model.setup(
        opt
    )  # regular setup: load and print networks; create schedulers
    cyclegan_model.eval()
    cyclegan_options = opt

    print("Loaded cyclegan model")
    print(cyclegan_model.device)

    input_nc = (
        cyclegan_options.output_nc
        if cyclegan_options.direction == "BtoA"
        else cyclegan_options.input_nc
    )
    transform = get_transform(opt=cyclegan_options, grayscale=(input_nc == 1))

    image_size = image_size_after_crop(
        env_name=BEAMNG_SIM_NAME,
        original_image_size=cyclegan_options.original_image_size,
    )

    for i, data in enumerate(dataset):

        cyclegan_model.set_input(data)  # unpack data from data loader
        start_time = time.perf_counter()
        cyclegan_model.test()  # run inference
        print("Inference time: {:.2f}s".format(time.perf_counter() - start_time))

        start_time = time.perf_counter()
        im_data = cyclegan_model.get_current_visuals().get("fake")
        # print(im_data.size())
        im = tensor2im(im_data)
        # print(im.shape)
        pil_image = Image.fromarray(im)
        # for some reason PIL wants the size parameter reversed
        pil_image = pil_image.resize(size=reversed(image_size))
        im = np.asarray(pil_image)
        # print(im.shape)
        # assert False
        # im = tensor2im(im_data)
        # print("Back and forth time: {:.2f}s".format(time.perf_counter() - start_time))
