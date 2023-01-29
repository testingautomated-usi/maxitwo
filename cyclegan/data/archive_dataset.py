import os.path

from PIL import Image

from config import SIMULATOR_NAMES
from cyclegan.data.base_dataset import BaseDataset, get_transform
from utils.dataset_utils import crop, load_archive


class ArchiveDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        assert not opt.isTrain, "Archive dataset not supported during training"
        assert opt.archive_filepath is not None, "archive_filepath not specified"
        assert os.path.exists(opt.archive_filepath), "archive_filepath {} does not exist".format(opt.archive_filepath)
        archive_path = opt.archive_filepath[: opt.archive_filepath.rindex(os.sep)]
        archive_name = opt.archive_filepath[opt.archive_filepath.rindex(os.sep) + 1 :]
        numpy_dict = load_archive(archive_path=archive_path, archive_name=archive_name)
        self.observations = numpy_dict["observations"]
        self.actions = numpy_dict["actions"]
        self.env_name = None
        for sim_name in SIMULATOR_NAMES:
            if sim_name in archive_name:
                self.env_name = sim_name
                break
        assert self.env_name is not None, "Env name {} not found in {}".format(SIMULATOR_NAMES, archive_name)

        input_nc = self.opt.output_nc if self.opt.direction == "BtoA" else self.opt.input_nc
        self.transform = get_transform(opt, grayscale=(input_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        obs = self.observations[index]
        obs_cropped = crop(image=obs, env_name=self.env_name)
        obs_pil = Image.fromarray(obs_cropped)
        obs_tensor = self.transform(obs_pil)

        return {"A": obs_tensor, "A_paths": [], "actions": self.actions[index]}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.observations)
