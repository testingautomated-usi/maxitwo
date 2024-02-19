from cyclegan.options.base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument(
            "--results_dir", type=str, default="./results/", help="saves results here."
        )
        parser.add_argument(
            "--aspect_ratio",
            type=float,
            default=1.0,
            help="aspect ratio of result images",
        )
        parser.add_argument(
            "--phase", type=str, default="test", help="train, val, test, etc"
        )
        # Dropout and Batchnorm has different behaviour during training and test.
        parser.add_argument(
            "--eval", action="store_true", help="use eval mode during test time."
        )
        parser.add_argument(
            "--num_test",
            type=int,
            default=-1,
            help="how many test images to run, default all",
        )
        parser.add_argument(
            "--save_fake_only",
            action="store_true",
            default=False,
            help="only saving fake images",
        )
        parser.add_argument(
            "--archive_filepath",
            type=str,
            default=None,
            help="path to archive with images",
        )
        parser.add_argument(
            "--original_image_size",
            type=tuple,
            default=(140, 320),
            help="path to archive with images",
        )
        # rewrite devalue values
        parser.set_defaults(model="test")
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default("crop_size"))
        self.isTrain = False
        return parser
