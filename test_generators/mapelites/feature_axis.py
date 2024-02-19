import numpy as np


# Axis of the feature map
class FeatureAxis:

    def __init__(self, bins: int, name: str):
        self.bins = bins
        self.min = np.inf
        self.name = name
