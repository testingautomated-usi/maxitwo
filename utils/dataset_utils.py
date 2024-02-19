import os
from typing import Tuple, List, Dict

import cv2
import numpy as np
from tensorflow import keras

from config import (
    SIMULATOR_NAMES,
    BEAMNG_SIM_NAME,
    UDACITY_SIM_NAME,
    DONKEY_SIM_NAME,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    TEST_SPLIT_BEAMNG,
    TEST_SPLIT_DONKEY,
    TEST_SPLIT_UDACITY,
)
from global_log import GlobalLog
from self_driving.road import Road
from sklearn.model_selection import train_test_split


def save_archive(
    actions: List[np.ndarray],
    observations: List[np.ndarray],
    is_success_flags: List[bool],
    car_positions_x_episodes: List[List[float]],
    car_positions_y_episodes: List[List[float]],
    episode_lengths: List[int],
    tracks: List[Road],
    archive_path: str,
    archive_name: str,
) -> None:

    os.makedirs(archive_path, exist_ok=True)

    logg = GlobalLog("save_archive")

    actions_array = np.array(actions)
    observations_array = np.array(observations)
    is_success_flags_array = np.asarray(is_success_flags)
    tracks_concrete = []
    tracks_control_points = []
    for track in tracks:
        tracks_concrete.append(track.get_concrete_representation(to_plot=True))
        tracks_control_points.append(
            [(cp.x, cp.y, cp.z) for cp in track.control_points]
        )

    if "supervised" not in archive_name:
        assert len(observations_array) == len(
            actions_array
        ), "The two arrays should have the same length. Actions length: {}. Observations length: {}".format(
            len(actions_array), len(observations_array)
        )

    numpy_dict = {
        "actions": actions_array,
        "observations": observations_array,
        "is_success_flags": is_success_flags_array,
        "tracks_concrete": np.asarray(tracks_concrete),
        "tracks_control_points": np.asarray(tracks_control_points),
        "car_positions_x_episodes": np.asarray(car_positions_x_episodes),
        "car_positions_y_episodes": np.asarray(car_positions_y_episodes),
        "episode_lengths": np.asarray(episode_lengths),
    }

    logg.info("Actions: {}".format(numpy_dict["actions"].shape))
    logg.info("Observations: {}".format(numpy_dict["observations"].shape))
    logg.info("Is success flags: {}".format(numpy_dict["is_success_flags"].shape))
    logg.info("Tracks concrete: {}".format(numpy_dict["tracks_concrete"].shape))
    logg.info(
        "Tracks control points: {}".format(numpy_dict["tracks_control_points"].shape)
    )
    logg.info(
        "Car positions x for each episode: {}".format(
            numpy_dict["car_positions_x_episodes"].shape
        )
    )
    logg.info(
        "Car positions y for each episode: {}".format(
            numpy_dict["car_positions_y_episodes"].shape
        )
    )
    logg.info("Episode lengths: {}".format(numpy_dict["episode_lengths"].shape))

    np.savez(os.path.join(archive_path, "{}.npz".format(archive_name)), **numpy_dict)


def _load_numpy_archive(archive_path: str, archive_name: str) -> Dict:
    assert os.path.exists(
        os.path.join(archive_path, archive_name)
    ), "Archive file {} does not exist".format(os.path.join(archive_path, archive_name))

    # https://stackoverflow.com/questions/55890813/how-to-fix-object-arrays-cannot-be-loaded-when-allow-pickle-false-for-imdb-loa

    # save np.load
    # np_load_old = np.load

    # modify the default parameters of np.load
    # np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

    return np.load(os.path.join(archive_path, archive_name), allow_pickle=True)
    # return np.load(os.path.join(archive_path, archive_name))


def load_archive(archive_path: str, archive_name: str) -> Dict:
    return _load_numpy_archive(archive_path=archive_path, archive_name=archive_name)


def load_archive_into_dataset(
    archive_path: str,
    archive_names: List[str],
    seed: int,
    test_split: float = 0.2,
    custom_test_split: bool = False,
    predict_throttle: bool = False,
    env_name: str = None,
    percentage_data: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    logg = GlobalLog("load_archive_into_dataset")

    if env_name == "mixed":
        X_train, X_test, y_train, y_test = [], [], [], []
        for sim_name in SIMULATOR_NAMES:
            filtered_archive_names = list(
                filter(lambda an: sim_name in an, archive_names)
            )
            assert (
                len(filtered_archive_names) <= 1
            ), "There must be at most one archive name that contains {}. Found: {}".format(
                sim_name, filtered_archive_names
            )

            if len(filtered_archive_names) == 1:

                filtered_archive_name = filtered_archive_names[0]
                numpy_dict = load_archive(
                    archive_path=archive_path, archive_name=filtered_archive_name
                )
                obs = numpy_dict["observations"]
                actions = numpy_dict["actions"]

                if len(actions.shape) > 2:
                    actions = actions.squeeze(axis=1)

                if not predict_throttle:
                    actions = actions[:, 0]

                if custom_test_split:

                    if sim_name == BEAMNG_SIM_NAME:
                        _test_split = TEST_SPLIT_BEAMNG

                    elif sim_name == UDACITY_SIM_NAME:
                        _test_split = TEST_SPLIT_UDACITY

                    elif sim_name == DONKEY_SIM_NAME:
                        _test_split = TEST_SPLIT_DONKEY

                    else:
                        raise NotImplementedError(f"Unknown env name: {sim_name}")

                    logg.info(f"Using custom test split for {sim_name}: {_test_split}")
                else:
                    _test_split = test_split

                X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
                    obs, actions, test_size=_test_split, random_state=seed
                )
                logg.info(
                    "Training set size for {}: {}. Validation set size for {}: {}".format(
                        sim_name, X_train_i.shape, sim_name, X_test_i.shape
                    )
                )

                if percentage_data < 1.0:
                    length_dataset = len(X_train_i)
                    length_dataset_to_consider = int(percentage_data * length_dataset)
                    items_to_remove = length_dataset - length_dataset_to_consider
                    indices_to_remove = np.random.choice(
                        a=np.arange(start=0, stop=items_to_remove),
                        size=items_to_remove,
                        replace=False,
                    )
                    X_train_i = np.delete(X_train_i, indices_to_remove, axis=0)
                    y_train_i = np.delete(y_train_i, indices_to_remove, axis=0)

                    logg.info(
                        f"Training set size for {sim_name} after filtering: {X_train_i.shape}."
                    )

                logg.info("Adding sim name dimension")
                X_train.append([X_train_i, sim_name])
                X_test.append([X_test_i, sim_name])
                y_train.append(y_train_i)
                y_test.append(y_test_i)

        assert len(X_train) > 0, "Training data must not be empty"

        X_train = np.concatenate(X_train)
        X_test = np.concatenate(X_test)
        y_train = np.concatenate(y_train)
        y_test = np.concatenate(y_test)

        logg.info(
            "Mixed training set size: {}. Mixed validation set size: {}".format(
                X_train.shape, X_test.shape
            )
        )

        return X_train, X_test, y_train, y_test

    obs = []
    actions = []
    for i in range(len(archive_names)):
        numpy_dict = load_archive(
            archive_path=archive_path, archive_name=archive_names[i]
        )
        obs_i = numpy_dict["observations"]
        actions_i = numpy_dict["actions"]
        obs.append(obs_i)
        actions.append(actions_i)

    obs = np.concatenate(obs)
    actions = np.concatenate(actions)

    X = obs

    if len(actions.shape) > 2:
        actions = actions.squeeze(axis=1)

    if not predict_throttle:
        y = actions[:, 0]
    else:
        y = actions

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=seed
    )
    logg.info(
        "Training set size: {}. Validation set size: {}".format(
            X_train.shape, X_test.shape
        )
    )

    return X_train, X_test, y_train, y_test


# FIXME: this method used to modify an image taken randomly from the dataset; to me it makes sense to augment
#  the current image
def augment(
    image: np.ndarray,
    y: np.ndarray,
    range_x: int = 100,
    range_y: int = 10,
    fake_images: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate an augmented image and adjust steering angle.
    """
    image, y = random_flip(image=image, y=y)
    image, y = random_translate(image=image, y=y, range_x=range_x, range_y=range_y)
    if not fake_images:
        image = random_shadow(image=image)
    image = random_brightness(image=image)
    return image, y


def image_size_after_crop(env_name: str, original_image_size: Tuple) -> Tuple:
    # Depends on the crop method below

    if env_name == BEAMNG_SIM_NAME:
        return original_image_size[0] - 81, original_image_size[1]

    if env_name == UDACITY_SIM_NAME:
        return original_image_size[0] - 85, original_image_size[1]

    if env_name == DONKEY_SIM_NAME:
        return original_image_size[0] - 60, original_image_size[1]

    raise RuntimeError("Unknown simulator name: {}".format(env_name))


def crop(image: np.ndarray, env_name: str) -> np.ndarray:
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    if env_name == BEAMNG_SIM_NAME:
        return image[80:-1, :, :]  # remove the sky and the car front (from DeepJanus)

    if env_name == UDACITY_SIM_NAME:
        return image[60:-25, :, :]  # remove the sky and the car front (from Selforacle)

    if env_name == DONKEY_SIM_NAME:
        return image[60:, :, :]  # remove the sky

    raise RuntimeError("Unknown simulator name: {}".format(env_name))


def resize(image: np.ndarray) -> np.ndarray:
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def bgr2yuv(image: np.ndarray) -> np.ndarray:
    """
    Convert the image from BGR to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2YUV)


def preprocess(image: np.ndarray, env_name: str, fake_images: bool) -> np.ndarray:
    """
    Combine all preprocess functions into one
    """
    if not fake_images:
        image = crop(image=image, env_name=env_name)
    image = resize(image=image)
    image = bgr2yuv(image=image)
    return image


def random_flip(image: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly flip the image left <-> right, and adjust the label.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(src=image, flipCode=1)
        y[0] = -y[0]  # assume the first position is the steering angle
    return image, y


def random_translate(
    image: np.ndarray, y: np.ndarray, range_x: int, range_y: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly shift the image vertically and horizontally (translation).
    """
    if np.random.rand() < 0.5:
        trans_x = range_x * (np.random.rand() - 0.5)
        trans_y = range_y * (np.random.rand() - 0.5)
        # translate steering angle
        y[0] += trans_x * 0.002
        trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
        height, width = image.shape[:2]
        image = cv2.warpAffine(image, trans_m, (width, height))
    return image, y


def random_shadow(image: np.ndarray) -> np.ndarray:
    """
    Generates and adds random shadow
    """
    if np.random.rand() < 0.5:
        # (x1, y1) and (x2, y2) forms a line
        # xm, ym gives all the locations of the image
        x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
        x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
        xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

        # mathematically speaking, we want to set 1 below the line and zero otherwise
        # Our coordinate is up side down.  So, the above the line:
        # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
        # as x2 == x1 causes zero-division problem, we'll write it in the below form:
        # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
        mask = np.zeros_like(image[:, :, 1])
        mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

        # choose which side should have shadow and adjust saturation
        cond = mask == np.random.randint(2)
        s_ratio = np.random.uniform(low=0.2, high=0.5)

        # adjust Saturation in HLS(Hue, Light, Saturation)
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
        return cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)
    return image


def random_brightness(image: np.ndarray) -> np.ndarray:
    """
    Randomly adjust brightness of the image.
    """
    if np.random.rand() < 0.5:
        # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
        hsv[:, :, 2] = hsv[:, :, 2] * ratio
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image


class DataGenerator(keras.utils.Sequence):

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        is_training: bool,
        env_name: str,
        input_shape: Tuple[int],
        predict_throttle: bool = False,
        preprocess: bool = True,
        fake_images: bool = False,
        no_augment: bool = False,
    ):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.is_training = is_training
        self.env_name = env_name
        if env_name == "mixed":
            # data format is X = [(images, sim_name_1), (images, sim_name_2),..., (images, sim_name_N)]
            num_all_data = sum([X[i].shape[0] for i in range(0, len(X), 2)])
            self.indexes = np.arange(num_all_data)
        else:
            self.indexes = np.arange(len(X))
        self.input_shape = input_shape
        self.predict_throttle = predict_throttle
        self.preprocess = preprocess
        self.fake_images = fake_images
        self.no_augment = no_augment
        self.on_epoch_end()

        # allow for mixed datasets
        assert (
            env_name in SIMULATOR_NAMES or env_name == "mixed"
        ), "Unknown simulator name {}. Choose among {}".format(
            env_name, SIMULATOR_NAMES
        )

    def on_epoch_end(self) -> None:
        """Updates indexes after each epoch"""
        if self.is_training:
            np.random.shuffle(self.indexes)

    def get_index_and_env_name_mixed_data(self, idx: int) -> Tuple[np.ndarray, str]:
        # data format is X = [(images, sim_name_1), (images, sim_name_2),..., (images, sim_name_N)]
        start, end = 0, 0
        for i in range(0, len(self.X), 2):
            if start <= idx < end + len(self.X[i]):
                _idx = idx - end
                return self.X[i][_idx], self.X[i + 1]
            start = end + len(self.X[i])
            end += len(self.X[i])

        raise RuntimeError("Index {} not present in data".format(idx))

    def get_data_batch(self, batch_indexes: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Generates data containing batch_size samples"""

        X_batch = np.empty((self.batch_size, *self.input_shape))
        if self.predict_throttle:
            y_batch = np.empty((self.batch_size, 2))
        else:
            y_batch = np.empty(self.batch_size)

        for i, idx in enumerate(batch_indexes):

            env_name = None
            if self.env_name == "mixed":
                X_item, env_name = self.get_index_and_env_name_mixed_data(idx=idx)
            else:
                X_item = self.X[idx]

            y_item = self.y[idx]

            if type(y_item) == np.float64 or type(y_item) == np.float32:
                y_item = np.asarray([y_item])

            if self.is_training and np.random.rand() < 0.5 and not self.no_augment:
                X_item, y_item = augment(
                    image=X_item, y=y_item, fake_images=self.fake_images
                )

            if self.preprocess:
                if self.env_name == "mixed":
                    assert (
                        env_name is not None
                    ), "Env name for mixed dataset preprocessing not assigned"
                    X_batch[i] = preprocess(
                        image=X_item, env_name=env_name, fake_images=self.fake_images
                    )
                else:
                    assert (
                        self.env_name in SIMULATOR_NAMES
                    ), "Unknown simulator name: {}".format(self.env_name)
                    X_batch[i] = preprocess(
                        image=X_item,
                        env_name=self.env_name,
                        fake_images=self.fake_images,
                    )

            y_batch[i] = y_item

        return X_batch, y_batch

    def __len__(self):
        """Denotes the number of batches per epoch"""

        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        """Generates one batch of data"""

        batch_indexes = self.indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        batch_indexes = [self.indexes[batch_idx] for batch_idx in batch_indexes]
        X_batch, y_batch = self.get_data_batch(batch_indexes=batch_indexes)

        return X_batch, y_batch


if __name__ == "__main__":

    train_data, test_data, train_labels, test_labels = load_archive_into_dataset(
        archive_path="../logs",
        archive_names=[
            "beamng-short.npz",
            "donkey-short.npz",
            "udacity-short.npz",
        ],
        seed=0,
        test_split=0.2,
        predict_throttle=False,
        env_name="mixed",
    )
    print(train_data.shape, test_data.shape)
    print(train_labels.shape, test_labels.shape)
