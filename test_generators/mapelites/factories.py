from typing import Dict, List, Tuple, Union

import numpy as np

from test_generators.mapelites.config import (
    CURVATURE_FEATURE_NAME,
    MAX_LATERAL_POSITION_METRIC,
    OLD_STD_STEERING_ANGLE_FEATURE_NAME,
    STD_LATERAL_POSITION_METRIC,
    STD_SPEED_METRIC,
    STD_STEERING_ANGLE_FEATURE_NAME,
    STD_STEERING_ANGLE_METRIC,
    TURNS_COUNT_FEATURE_NAME,
)
from test_generators.mapelites.curvature_feature import CurvatureFeature
from test_generators.mapelites.feature import Feature
from test_generators.mapelites.individual import Individual
from test_generators.mapelites.steering_angle_feature import StdSteeringAngleFeature
from test_generators.mapelites.turns_count_feature import TurnsCountFeature


def make_features_given_concrete_features(concrete_features: List[Tuple[str, Union[int, float]]]) -> List[Feature]:
    features = []
    for concrete_feature in concrete_features:
        feature_name, feature_value = concrete_feature[0], concrete_feature[1]
        if feature_name == STD_STEERING_ANGLE_FEATURE_NAME or feature_name == OLD_STD_STEERING_ANGLE_FEATURE_NAME:
            features.append(StdSteeringAngleFeature(steering_angles=[], steering_angle_bin=feature_value))
        elif feature_name == CURVATURE_FEATURE_NAME:
            features.append(CurvatureFeature(curvature=None, curvature_bin=feature_value))
        elif feature_name == TURNS_COUNT_FEATURE_NAME:
            features.append(TurnsCountFeature(turns_count=None, turns_count_bin=feature_value))
        else:
            raise NotImplemented("Feature {} not supported".format(feature_name))
    return features


def make_features_given_feature_combination(
    feature_combination: str, steering_angles: List[float], individual: Individual, mock: bool = False
) -> Tuple[Feature, Feature]:
    if feature_combination == "{}-{}".format(STD_STEERING_ANGLE_FEATURE_NAME, CURVATURE_FEATURE_NAME):
        return (
            StdSteeringAngleFeature(steering_angles=steering_angles, mock=mock),
            CurvatureFeature(curvature=individual.get_representation().compute_curvature())
            if not mock
            else CurvatureFeature(curvature=None, mock=mock),
        )
    if feature_combination == "{}-{}".format(TURNS_COUNT_FEATURE_NAME, CURVATURE_FEATURE_NAME):
        num_turns = individual.get_representation().compute_num_turns()[0]
        return (
            TurnsCountFeature(turns_count=num_turns) if not mock else TurnsCountFeature(turns_count=None, mock=mock),
            CurvatureFeature(curvature=individual.get_representation().compute_curvature())
            if not mock
            else CurvatureFeature(curvature=None, mock=mock),
        )
    raise NotImplemented("Feature combination {} not supported".format(feature_combination))


def make_behavioural_metrics_given_dict(
    individual_dict: Dict,
) -> Dict:
    names = Individual.get_behavioural_metrics_names()
    behavioural_metrics = dict()
    for name in names:
        if name in individual_dict:
            behavioural_metrics[name] = individual_dict[name]
    return behavioural_metrics


def make_quality_metric(quality_metric: str, behavioural_metric_name: str, behavioural_metrics: Dict) -> Union[float, int]:
    values = behavioural_metrics[behavioural_metric_name]
    if (
        quality_metric == STD_STEERING_ANGLE_METRIC
        or quality_metric == STD_LATERAL_POSITION_METRIC
        or quality_metric == STD_SPEED_METRIC
    ):
        return float(np.std(values))
    if quality_metric == MAX_LATERAL_POSITION_METRIC:
        # actually what is logged is the out of bound distance
        array_values = np.asarray(values)
        return np.max(2 - np.clip(array_values, a_min=-0.1, a_max=np.max(array_values)))
    raise NotImplementedError("Unknown quality metric: {}".format(quality_metric))
