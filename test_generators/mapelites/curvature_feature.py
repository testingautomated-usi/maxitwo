from typing import Union

import numpy as np

from test_generators.mapelites.config import CURVATURE_FEATURE_NAME
from test_generators.mapelites.feature import Feature


class CurvatureFeature(Feature):
    def __init__(self, curvature: float, curvature_bin: int = None, mock: bool = False):
        super(CurvatureFeature, self).__init__(feature_bin=curvature_bin)
        self.name = CURVATURE_FEATURE_NAME
        self.curvature = curvature
        self.mock = mock
        self.mock_value = self.get_random_value()

    def get_value(self) -> Union[float, int]:
        if not self.mock:
            assert (
                self.curvature is not None or self.feature_bin is not None
            ), "Get value for curvature feature needs either a curvature value or a curvature bin"
            if self.curvature is not None:
                # FIXME: compute curvature here given road
                # * 100 trick for binning different values in different cells
                return int(self.curvature * 100)
            return self.feature_bin
        return self.mock_value

    def get_random_value(self) -> Union[float, int]:
        return np.random.randint(low=0, high=50)
