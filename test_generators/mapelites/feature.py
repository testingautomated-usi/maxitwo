from abc import ABC, abstractmethod
from typing import Union


class Feature(ABC):

    def __init__(self, feature_bin: None):
        self.name = None
        self.feature_bin = feature_bin

    @abstractmethod
    def get_value(self) -> Union[float, int]:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def get_random_value(self) -> Union[float, int]:
        raise NotImplementedError("Not implemented")
