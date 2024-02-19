from typing import Union, List

import numpy as np

from test_generators.mapelites.config import STD_STEERING_ANGLE_FEATURE_NAME
from test_generators.mapelites.feature import Feature


class StdSteeringAngleFeature(Feature):

    def __init__(
        self,
        steering_angles: List[float],
        steering_angle_bin: int = None,
        mock: bool = False,
    ):
        super(StdSteeringAngleFeature, self).__init__(feature_bin=steering_angle_bin)
        self.name = STD_STEERING_ANGLE_FEATURE_NAME
        self.steering_angles = steering_angles
        self.mock = mock
        self.mock_value = self.get_random_value()

    def get_value(self) -> Union[float, int]:
        if not self.mock:
            assert (
                self.steering_angles is not None and len(self.steering_angles) > 0
            ) or self.feature_bin is not None, "Get steering angle needs either the list of steering angles or a steering angle value"

            if self.steering_angles is not None and len(self.steering_angles) > 0:
                # casting to int for binning values into cells
                # * 100 trick for binning different values in different cells
                return int(np.std(self.steering_angles) * 100)
            return self.feature_bin
        return self.mock_value

    def get_random_value(self) -> Union[float, int]:
        return np.random.randint(low=0, high=100)
