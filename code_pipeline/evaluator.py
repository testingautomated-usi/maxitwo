from abc import ABC, abstractmethod

from test_generators.mapelites.individual import Individual


class Evaluator(ABC):
    def __init__(self, feature_combination: str):
        self.feature_combination = feature_combination

    @abstractmethod
    def run_sim(self, individual: Individual) -> None:
        raise NotImplemented("Not implemented")

    @abstractmethod
    def close(self) -> None:
        raise NotImplemented("Not implemented")
