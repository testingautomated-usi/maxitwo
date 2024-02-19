import numpy as np

from code_pipeline.evaluator import Evaluator
from global_log import GlobalLog
from test_generators.mapelites.curvature_feature import CurvatureFeature
from test_generators.mapelites.factories import make_features_given_feature_combination
from test_generators.mapelites.individual import Individual
from test_generators.mapelites.lateral_position_fitness import LateralPositionFitness
from test_generators.mapelites.steering_angle_feature import StdSteeringAngleFeature


class MockEvaluator(Evaluator):

    def __init__(self, feature_combination: str):
        super(MockEvaluator, self).__init__(feature_combination=feature_combination)
        self.logger = GlobalLog("mock_evaluator")

    def run_sim(self, individual: Individual) -> None:

        self.logger.info('Executing individual with id {}'.format(individual.id))

        # TODO: possibly add a sleep

        # set fitness and all possible features
        individual.set_fitness(LateralPositionFitness(lateral_positions=[], mock=True))
        individual.set_features(
            features=make_features_given_feature_combination(
                feature_combination=self.feature_combination,
                steering_angles=[],
                individual=individual,
            )
        )
        nums = np.random.randint(low=5, high=100)
        individual.set_behavioural_metrics(
            speeds=list(np.random.uniform(low=0.0, high=35.0, size=(nums,))),
            steering_angles=list(np.random.uniform(low=-1.0, high=1.0, size=(nums,))),
            lateral_positions=list(np.random.uniform(low=-0.1, high=2.0, size=(nums,))),
        )

    def close(self) -> None:
        pass
