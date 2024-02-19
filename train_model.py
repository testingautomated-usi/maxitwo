import argparse
import os

import numpy as np

from autopilot_model import AutopilotModel
from config import SIMULATOR_NAMES, INPUT_SHAPE, DAVE2_NAME, MODEL_NAMES
from global_log import GlobalLog
from utils.dataset_utils import load_archive_into_dataset
from utils.randomness import set_random_seed

parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="Random seed", type=int, default=-1)
parser.add_argument("--archive-path", help="Archive path", type=str, default="logs")
parser.add_argument(
    "--env-name",
    help="Simulator name",
    type=str,
    choices=[*SIMULATOR_NAMES, "mixed"],
    required=True,
)
parser.add_argument(
    "--archive-names",
    nargs="+",
    help="Archive name to analyze (with extension, .npz)",
    type=str,
    required=True,
)
parser.add_argument(
    "--model-save-path",
    help="Path where model will be saved",
    type=str,
    default=os.path.join("logs", "models"),
)
parser.add_argument(
    "--model-name",
    help="Model name (without the extension)",
    choices=MODEL_NAMES,
    type=str,
    default=DAVE2_NAME,
    required=True,
)
parser.add_argument(
    "--predict-throttle",
    help="Predict steering and throttle",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--no-preprocess",
    help="Do not preprocess data during training",
    action="store_true",
    default=False,
)
parser.add_argument("--test-split", help="Test split", type=float, default=0.2)
parser.add_argument(
    "--keep-probability", help="Keep probability (dropout)", type=float, default=0.5
)
parser.add_argument("--learning-rate", help="Learning rate", type=float, default=1e-4)
parser.add_argument("--nb-epoch", help="Number of epochs", type=int, default=200)
parser.add_argument("--batch-size", help="Batch size", type=int, default=128)
parser.add_argument(
    "--early-stopping-patience",
    help="Number of epochs of no validation loss improvement used to stop training",
    type=int,
    default=3,
)
parser.add_argument(
    "--fake-images",
    help="Whether the training is performed on images produced by the cyclegan. The fake images contained on the archives are already cropped.",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--no-augment", help="Do not apply augmentation", action="store_true", default=False
)
parser.add_argument(
    "--custom-test-split",
    help="Whether or not to use a custom test split for each env (in config) and env_name == mixed",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--percentage-data",
    help="Percentage of data to use for training (1.0 means the whole dataset).",
    type=float,
    choices=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    default=1.0,
)
parser.add_argument(
    "--decay-learning-rate",
    help="Whether or not to decay the learning rate",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--decay-learning-rate-after-epoch",
    help="Number of epochs after which learning rate is decayed",
    type=int,
    default=10,
)
parser.add_argument(
    "--learning-rate-decay-rate",
    help="Exponent of the learning rate exponential decay operator.",
    type=float,
    default=0.1,
)
args = parser.parse_args()

if __name__ == "__main__":

    logg = GlobalLog("train_model")

    if args.seed == -1:
        args.seed = np.random.randint(2**30 - 1)

    logg.info("Random seed: {}".format(args.seed))
    set_random_seed(seed=args.seed)

    train_data, test_data, train_labels, test_labels = load_archive_into_dataset(
        archive_path=args.archive_path,
        archive_names=args.archive_names,
        seed=args.seed,
        test_split=args.test_split,
        custom_test_split=args.custom_test_split,
        predict_throttle=args.predict_throttle,
        env_name=None if args.env_name != "mixed" else "mixed",
        percentage_data=args.percentage_data,
    )

    autopilot_model = AutopilotModel(
        env_name=args.env_name,
        input_shape=INPUT_SHAPE,
        predict_throttle=args.predict_throttle,
    )

    autopilot_model.train_model(
        X_train=train_data,
        X_val=test_data,
        y_train=train_labels,
        y_val=test_labels,
        save_path=args.model_save_path,
        model_name=args.model_name,
        save_best_only=True,
        keep_probability=args.keep_probability,
        learning_rate=args.learning_rate,
        nb_epoch=args.nb_epoch,
        batch_size=args.batch_size,
        early_stopping_patience=args.early_stopping_patience,
        save_plots=True,
        preprocess=not args.no_preprocess,
        fake_images=args.fake_images,
        no_augment=args.no_augment,
        decay_learning_rate=args.decay_learning_rate,
        decay_learning_rate_after_epochs=args.decay_learning_rate_after_epoch,
        learning_rate_decay_rate=args.learning_rate_decay_rate,
    )
