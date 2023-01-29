from typing import List, Tuple

import numpy as np


def _get_cohend_thresholds(effect_size: float) -> str:
    magnitude = "large"
    if abs(effect_size) < 0.50:
        magnitude = "small"
    elif 0.50 <= abs(effect_size) < 0.80:
        magnitude = "medium"

    return magnitude


# function to calculate Cohen's d for independent samples
def cohend(a: List[float], b: List[float]) -> Tuple[float, str]:

    if type(a) == list:
        a = np.asarray(a)
    if type(b) == list:
        b = np.asarray(b)

    # calculate the size of samples
    m, n = len(a), len(b)
    # assert m == n, "The two list must be of the same length: {}, {}".format(m, n) # FIXME: check
    # calculate the variance of the samples
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    # calculate the pooled standard deviation
    s = np.sqrt(((m - 1) * s1 + (n - 1) * s2) / (m + n - 2))
    # calculate the means of the samples
    u1, u2 = np.mean(a), np.mean(b)
    # calculate the effect size
    effect_size = (u1 - u2) / s
    return effect_size, _get_cohend_thresholds(effect_size=effect_size)
