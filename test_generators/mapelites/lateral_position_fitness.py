from typing import List

import numpy as np

from test_generators.mapelites.fitness import Fitness


class LateralPositionFitness(Fitness):
    def __init__(self, lateral_positions: List[float], min_lateral_position: float = None, mock: bool = False):
        super(LateralPositionFitness, self).__init__()
        self.lateral_positions = lateral_positions
        self.min_lateral_position = min_lateral_position
        self.mock = mock
        self.name = "lateral_position_fitness"
        self.mock_value = self.get_random_value()

    def get_value(self) -> float:
        if not self.mock:
            if self.min_lateral_position is not None:
                return self.min_lateral_position
            assert len(self.lateral_positions) > 0, "List of lateral positions cannot be empty"
            return round(min(self.lateral_positions), 4) if round(min(self.lateral_positions), 4) > 0 else -0.1
        return self.mock_value

    def get_min_value(self) -> float:
        return -0.1

    def get_max_value(self) -> float:
        return 2.0

    def get_random_value(self) -> float:
        return round(np.random.uniform(low=-0.1, high=2), 4)
