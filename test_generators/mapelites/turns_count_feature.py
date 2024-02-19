from typing import Union

import numpy as np

from test_generators.mapelites.config import TURNS_COUNT_FEATURE_NAME
from test_generators.mapelites.feature import Feature


class TurnsCountFeature(Feature):

    def __init__(
        self, turns_count: int, turns_count_bin: int = None, mock: bool = False
    ):
        super(TurnsCountFeature, self).__init__(feature_bin=turns_count_bin)
        self.name = TURNS_COUNT_FEATURE_NAME
        self.turns_count = turns_count
        self.mock = mock
        self.mock_value = self.get_random_value()

    def get_value(self) -> Union[float, int]:
        if not self.mock:
            assert (
                self.turns_count is not None or self.feature_bin is not None
            ), "Get value for turns count feature needs either a num_turns value or a turns_count bin"
            if self.turns_count is not None:
                # FIXME: compute turns_count here given road
                return self.turns_count
            return self.feature_bin
        return self.mock_value

    def get_random_value(self) -> Union[float, int]:
        return np.random.randint(low=0, high=10)
