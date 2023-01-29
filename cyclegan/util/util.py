"""This module contains simple helper functions """
from __future__ import print_function

import os
from collections import namedtuple
from typing import Dict, NamedTuple, Union

import numpy as np
import torch
from PIL import Image

Opt = namedtuple(
    "Opt",
    [
        "model_suffix",
        "model",
        "no_flip",
        "no_dropout",
        "input_nc",
        "output_nc",
        "ngf",
        "ndf",
        "netD",
        "netG",
        "n_layers_D",
        "norm",
        "init_type",
        "init_gain",
        "num_threads",
        "batch_size",
        "serial_batches",
        "display_id",
        "name",
        "gpu_ids",
        "checkpoints_dir",
        "epoch",
        "isTrain",
        "preprocess",
        "load_iter",
        "verbose",
        "direction",
        "original_image_size",
        "load_size",
        "crop_size",
    ],
)


def get_base_and_test_default_options(name: str, gpu_ids: str, checkpoints_dir: str, epoch: Union[int, str]) -> NamedTuple:
    # set gpu ids
    str_ids = gpu_ids.split(",")
    gpu_ids = []
    for str_id in str_ids:
        int_id = int(str_id)
        if int_id >= 0:
            gpu_ids.append(int_id)
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])

    opt = Opt(
        model_suffix="_A",
        model="test",
        no_flip=True,
        no_dropout=True,
        input_nc=3,
        output_nc=3,
        ngf=64,
        ndf=64,
        netD="basic",
        netG="resnet_9blocks",
        n_layers_D=3,
        norm="instance",
        init_type="normal",
        init_gain=0.02,
        num_threads=0,
        batch_size=1,
        serial_batches=True,
        display_id=-1,
        name=name,
        gpu_ids=gpu_ids,
        checkpoints_dir=checkpoints_dir,
        epoch=epoch,
        isTrain=False,
        preprocess="resize_and_crop",
        load_iter=0,
        verbose=True,
        direction="AtoB",
        original_image_size=(140, 320),
        load_size=286,
        crop_size=256,
    )

    return opt


def tensor2im(input_image, imtype=np.uint8):
    """Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: transpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name="network"):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print("shape,", x.shape)
    if val:
        x = x.flatten()
        print(
            "mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f"
            % (np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x))
        )


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
