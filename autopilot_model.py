import datetime
import os
import tensorflow as tf
from typing import Tuple

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import (
    Lambda,
    Conv2D,
    Dropout,
    Flatten,
    Dense,
    MaxPooling2D,
)
from tensorflow.keras.optimizers import Adam

from config import CHAUFFEUR_NAME, DAVE2_NAME, EPOCH_NAME, SIMULATOR_NAMES, INPUT_SHAPE

import matplotlib.pyplot as plt

from utils.dataset_utils import DataGenerator
from tensorflow.keras.models import load_model


class AutopilotModel:

    def __init__(
        self,
        env_name: str,
        input_shape: Tuple[int] = INPUT_SHAPE,
        predict_throttle: bool = True,
    ):
        # cropped input_shape: height, width, channels. Allow for mixed datasets
        assert (
            env_name in SIMULATOR_NAMES or env_name == "mixed"
        ), "Unknown simulator name {}. Choose among {}".format(
            env_name, SIMULATOR_NAMES
        )
        self.input_shape = input_shape
        self.env_name = env_name
        self.predict_throttle = predict_throttle
        self.model = None

    # selforacle paper
    # def _build_epoch_model(self, keep_probability: float = 0.5) -> Sequential:
    #     model = Sequential()
    #     model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=self.input_shape))

    #     model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    #     model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #     model.add(Dropout(0.25))

    #     model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    #     model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #     model.add(Dropout(0.25))

    #     model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    #     model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #     model.add(Dropout(0.5))

    #     model.add(Flatten())
    #     model.add(Dense(1024, activation='relu'))
    #     model.add(Dropout(keep_probability))

    #     if self.predict_throttle:
    #         model.add(Dense(2))
    #     else:
    #         model.add(Dense(1))

    #     return model

    # mind the gap paper (half the size)
    def _build_epoch_model(self, keep_probability: float = 0.5) -> Sequential:
        model = Sequential()
        model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=self.input_shape))

        model.add(Conv2D(16, (3, 3), activation="relu", padding="same"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Dropout(0.1))

        model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Dropout(0.1))

        model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Dropout(0.1))

        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dropout(keep_probability))

        if self.predict_throttle:
            model.add(Dense(2))
        else:
            model.add(Dense(1))

        return model

    def _build_dave2_model(self, keep_probability: float = 0.5) -> Sequential:
        """
        Modified NVIDIA model
        """
        model = Sequential()
        model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=self.input_shape))
        model.add(Conv2D(24, (5, 5), activation="elu", strides=(2, 2)))
        model.add(Conv2D(36, (5, 5), activation="elu", strides=(2, 2)))
        model.add(Conv2D(48, (5, 5), activation="elu", strides=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation="elu"))
        model.add(Conv2D(64, (3, 3), activation="elu"))
        model.add(Dropout(keep_probability))
        model.add(Flatten())
        model.add(Dense(100, activation="elu"))
        model.add(Dense(50, activation="elu"))
        model.add(Dense(10, activation="elu"))

        if self.predict_throttle:
            model.add(Dense(2))
        else:
            model.add(Dense(1))

        return model

    def _build_chauffeur_model(self, keep_probability: float = 0.1) -> Sequential:

        # Taken from https://github.com/udacity/self-driving-car/blob/master/steering-models/community-models/chauffeur/models.py
        # use_adadelta = True
        # learning_rate = 0.01
        # W_l2 = 0.0001
        # input_shape = CHAUFFEUR_INPUT_SHAPE
        # input_shape = INPUT_SHAPE  # Original Chauffeur uses input_shape=(120, 320, 3)

        model = Sequential()
        model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=self.input_shape))
        model.add(
            Conv2D(
                16,
                (5, 5),
                input_shape=self.input_shape,
                kernel_initializer="he_normal",
                bias_initializer="he_normal",
                activation="relu",
                padding="same",
            )
        )
        model.add(Dropout(0.05))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(
            Conv2D(
                20,
                (5, 5),
                kernel_initializer="he_normal",
                bias_initializer="he_normal",
                activation="relu",
                padding="same",
            )
        )
        model.add(Dropout(0.05))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(
            Conv2D(
                40,
                (3, 3),
                kernel_initializer="he_normal",
                bias_initializer="he_normal",
                activation="relu",
                padding="same",
            )
        )
        model.add(Dropout(0.05))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(
            Conv2D(
                60,
                (3, 3),
                kernel_initializer="he_normal",
                bias_initializer="he_normal",
                activation="relu",
                padding="same",
            )
        )
        model.add(Dropout(0.05))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(
            Conv2D(
                80,
                (2, 2),
                kernel_initializer="he_normal",
                bias_initializer="he_normal",
                activation="relu",
                padding="same",
            )
        )
        model.add(Dropout(0.05))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(
            Conv2D(
                128,
                (2, 2),
                kernel_initializer="he_normal",
                bias_initializer="he_normal",
                activation="relu",
                padding="same",
            )
        )
        model.add(Flatten())
        model.add(Dropout(keep_probability))

        if self.predict_throttle:
            model.add(Dense(2))
        else:
            model.add(Dense(1))

        return model

    def build_model(self, model_name: str, keep_probability: float = 0.5) -> Sequential:
        if model_name == DAVE2_NAME:
            model = self._build_dave2_model(keep_probability=keep_probability)
        elif model_name == CHAUFFEUR_NAME:
            model = self._build_chauffeur_model(keep_probability=keep_probability)
        elif model_name == EPOCH_NAME:
            model = self._build_epoch_model(keep_probability=keep_probability)
        else:
            raise NotImplementedError(f"Unknown model name: {model_name}")

        model.summary()
        return model

    def load(self, model_path: str) -> None:
        assert os.path.exists(model_path), "Model path {} not found".format(model_path)
        with tf.device("/cpu:0"):
            self.model = load_model(filepath=model_path)

    def train_model(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        save_path: str,
        model_name: str,
        save_best_only: bool = True,
        keep_probability: float = 0.5,
        learning_rate: float = 1e-4,
        nb_epoch: int = 200,
        batch_size: int = 128,
        early_stopping_patience: int = 3,
        save_plots: bool = True,
        preprocess: bool = True,
        fake_images: bool = False,
        no_augment: bool = False,
        decay_learning_rate: bool = False,
        decay_learning_rate_after_epochs: int = 10,
        learning_rate_decay_rate: float = 0.1,
    ) -> None:
        os.makedirs(save_path, exist_ok=True)
        self.model = self.build_model(
            model_name=model_name, keep_probability=keep_probability
        )
        datetime_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filename = "{}-{}-{}.h5".format(self.env_name, model_name, datetime_str)
        if fake_images:
            filename = "{}-fake-{}-{}.h5".format(
                self.env_name, model_name, datetime_str
            )

        checkpoint = ModelCheckpoint(
            os.path.join(save_path, filename),
            monitor="val_loss",
            verbose=0,
            save_best_only=save_best_only,
            mode="auto",
        )

        self.model.compile(loss="mean_squared_error", optimizer=Adam(lr=learning_rate))
        # self.model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=lr_schedule))

        def scheduler(epoch: int, lr: float) -> float:
            if epoch < decay_learning_rate_after_epochs:
                return lr
            return lr * tf.math.exp(-learning_rate_decay_rate)

        learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
        print(f"Initial learning rate: {round(self.model.optimizer.lr.numpy(), 6)}")

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=early_stopping_patience
        )

        train_generator = DataGenerator(
            X=X_train,
            y=y_train,
            batch_size=batch_size,
            is_training=True,
            env_name=self.env_name,
            input_shape=self.input_shape,
            predict_throttle=self.predict_throttle,
            preprocess=preprocess,
            fake_images=fake_images,
            no_augment=no_augment,
        )
        validation_generator = DataGenerator(
            X=X_val,
            y=y_val,
            batch_size=batch_size,
            is_training=False,
            env_name=self.env_name,
            input_shape=self.input_shape,
            predict_throttle=self.predict_throttle,
            preprocess=preprocess,
            fake_images=fake_images,
        )

        history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=nb_epoch,
            use_multiprocessing=False,
            max_queue_size=10,
            workers=8,
            callbacks=(
                [checkpoint, early_stopping, learning_rate_scheduler]
                if decay_learning_rate
                else [checkpoint, early_stopping]
            ),
            verbose=1,
        )
        print(f"Final learning rate: {round(self.model.optimizer.lr.numpy(), 6)}")

        if save_plots:
            plt.figure()
            plt.plot(history.history["loss"])
            plt.plot(history.history["val_loss"])
            plt.title("model loss")
            plt.ylabel("loss")
            plt.xlabel("epoch")
            plt.legend(["train", "val"], loc="upper left")

            plt.savefig(
                os.path.join(
                    save_path,
                    (
                        f"{self.env_name}-loss-{datetime_str}-{model_name}.pdf"
                        if not fake_images
                        else f"{self.env_name}-loss-{datetime_str}-fake-{model_name}.pdf"
                    ),
                )
            )
