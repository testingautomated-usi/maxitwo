from abc import ABC, abstractmethod


class Fitness(ABC):

    def __init__(self):
        self.name = None

    @abstractmethod
    def get_value(self) -> float:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def get_min_value(self) -> float:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def get_max_value(self) -> float:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def get_random_value(self) -> float:
        raise NotImplementedError("Not implemented")
