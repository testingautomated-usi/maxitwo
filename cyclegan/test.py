"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import ntpath
import os
import time

import numpy as np
from PIL import Image

from config import SIMULATOR_NAMES
from cyclegan.data import create_dataset
from cyclegan.models import create_model
from cyclegan.options.test_options import TestOptions
from cyclegan.util import util
from utils.dataset_utils import image_size_after_crop

if __name__ == "__main__":
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    fake_obs = []
    actions = []
    for i, data in enumerate(dataset):
        if opt.num_test != -1 and i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        start_time = time.perf_counter()
        model.test()  # run inference
        print("Inference time: {:.2f}s. {}/{}".format(time.perf_counter() - start_time, i, len(dataset)))
        visuals = model.get_current_viøsuals()  # get image results
        img_path = model.get_image_paths()  # get image paths

        sim_name = None
        if opt.archive_filepath is not None:
            for s_name in SIMULATOR_NAMES:
                if s_name in opt.archive_filepath:
                    sim_name = s_name
                    break

        if len(img_path) > 0:
            image_dir = os.path.join(opt.dataroot[: opt.dataroot.rindex("/")], "translated{}".format(opt.model_suffix))
            os.makedirs(name=image_dir, exist_ok=True)

            short_path = ntpath.basename(img_path[0])
            name = os.path.splitext(short_path)[0]

            for label, im_data in visuals.items():
                im = util.tensor2im(im_data)
                image_name = "%s_%s.png" % (name, label)
                if (label == "real" and not opt.save_fake_only) or label == "fake":
                    save_path = os.path.join(image_dir, image_name)
                    util.save_image(im, save_path, aspect_ratio=opt.aspect_ratio)

        else:
            for label, im_data in visuals.items():
                im = util.tensor2im(im_data)
                image_name = "%s_%s.png" % ("test01", label)
                pil_image = Image.fromarray(im)
                if label == "fake":
                    assert sim_name is not None, "Sim name not assigned"
                    assert (
                        opt.original_image_size is not None and len(opt.original_image_size) > 0
                    ), "Original image size not assigned"
                    image_size = image_size_after_crop(env_name=sim_name, original_image_size=opt.original_image_size)
                    pil_image = Image.fromarray(im)
                    # for some reason PIL wants the size parameter reversed
                    pil_image = pil_image.resize(size=reversed(image_size))
                    im = np.asarray(pil_image)
                    fake_obs.append(im)
                    actions.append(data["actions"].to("cpu").detach().numpy())

    if len(fake_obs) > 0:
        archive_name = opt.archive_filepath[
            opt.archive_filepath.rindex("/") + 1 : opt.archive_filepath.rindex(".")
        ] + "-fake-{}.npz".format(opt.epoch)
        save_path = os.path.join("cyclegan", "datasets", archive_name)
        numpy_dict = {
            "actions": np.asarray(actions),
            "observations": np.asarray(fake_obs),
        }
        np.savez(save_path, **numpy_dict)
