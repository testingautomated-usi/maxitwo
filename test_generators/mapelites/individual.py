import copy
import random
from typing import List, Tuple, Set, Dict

import numpy as np

from self_driving.road import Road
from test_generators.mapelites.feature import Feature
from test_generators.mapelites.fitness import Fitness
from test_generators.mapelites.id_generator import IdGenerator


class Individual:

    def __init__(self, road: Road, start_id: int = 1):
        self.id = IdGenerator.get_instance(start_count=start_id).get_id()
        self.features: Tuple[Feature, Feature] = None
        self.fitness: Fitness = None
        self.representation: Road = road
        # FIXME: give a better name once it is clear what it means
        self.placed_mutant = 0
        self.selected_counter = 0
        self.observations = []
        self.behavioural_metrics = dict()

    def get_representation(self) -> Road:
        return self.representation

    def get_features(self) -> Tuple[Feature, Feature]:
        assert len(self.features) > 0, "Features have not been set"
        return self.features

    def get_fitness(self) -> Fitness:
        assert self.fitness is not None, "Fitness has not been set"
        return self.fitness

    def set_features(self, features: Tuple[Feature, Feature]) -> None:
        assert self.features is None, "Features already present"
        self.features = features

    def set_fitness(self, fitness: Fitness) -> None:
        self.fitness = fitness

    def get_behavioural_metrics(self) -> Dict:
        assert len(self.behavioural_metrics) > 0, "Behavioural metrics not set"
        return self.behavioural_metrics

    @staticmethod
    def get_behavioural_metrics_names() -> List[str]:
        # should synchronize with the method below
        return ["speeds", "steering_angles", "lateral_positions"]

    def set_behavioural_metrics_dict(self, behavioural_metrics: Dict) -> None:
        self.behavioural_metrics = behavioural_metrics

    def set_behavioural_metrics(
        self,
        speeds: List[float],
        steering_angles: List[float],
        lateral_positions: List[float],
    ) -> None:
        self.behavioural_metrics["speeds"] = speeds
        self.behavioural_metrics["steering_angles"] = steering_angles
        self.behavioural_metrics["lateral_positions"] = lateral_positions

    def set_observations(self, observations: List[np.ndarray]) -> None:
        self.observations = observations

    def get_observations(self) -> List[np.ndarray]:
        assert len(self.observations) > 0, "Observations not set"
        return self.observations

    @staticmethod
    # FIXME: quite cryptic
    def _next_gene_index(attempted_genes: Set, prefix_len: int, length: int) -> int:
        if len(attempted_genes) == length - 5:
            return -1
        i = random.randint(a=prefix_len, b=length - 1)
        j = 0
        while i in attempted_genes:
            j += 1
            i = random.randint(a=prefix_len, b=length - 1)
            if j > 1000000:
                raise RuntimeError(
                    "Mutation failed: attempted_genes: {}".format(attempted_genes)
                )

        attempted_genes.add(i)
        assert prefix_len <= i <= length - 1
        return i

    def mutate(self, mutation_extent: int, num_undo_attempts: int = 10) -> "Individual":

        # create new id for mutated individual
        mutated_individual = Individual(road=copy.deepcopy(self.representation))

        backup_control_points = list(mutated_individual.representation.control_points)
        attempted_genes = set()
        length = len(mutated_individual.representation.control_points) - 2

        gene_index = self._next_gene_index(
            attempted_genes=attempted_genes, prefix_len=4, length=length
        )

        while gene_index != -1:
            index_mutated, mut_value = mutated_individual.representation.mutate_gene(
                index=gene_index,
                lower_bound=-mutation_extent,
                upper_bound=mutation_extent,
            )

            attempt = 0

            is_valid = mutated_individual.representation.is_valid()
            while not is_valid and attempt < num_undo_attempts:
                mutated_individual.representation.undo_mutation(
                    gene_index, index_mutated, mut_value
                )
                index_mutated, mut_value = (
                    mutated_individual.representation.mutate_gene(
                        index=gene_index,
                        lower_bound=-mutation_extent,
                        upper_bound=mutation_extent,
                    )
                )
                attempt += 1
                is_valid = mutated_individual.representation.is_valid()

            if is_valid:
                break

            gene_index = self._next_gene_index(
                attempted_genes=attempted_genes, prefix_len=4, length=length
            )
            # print('Gene index: {}'.format(gene_index))

        if gene_index == -1:
            raise RuntimeError("No gene can be mutated")

        assert (
            mutated_individual.representation.is_valid()
        ), "Mutated individual is not valid"
        assert mutated_individual.representation.are_control_points_different(
            other_control_points=backup_control_points
        ), "All control points are equal after mutation"

        return mutated_individual

    def export(self) -> Dict:
        result = dict()
        result["id"] = self.id
        result["features"] = [
            (feature.name, feature.get_value()) for feature in self.features
        ]
        result["fitness"] = (self.fitness.name, self.fitness.get_value())
        result["representation"] = self.representation.export()
        if len(self.behavioural_metrics) > 0:
            for key, value in self.behavioural_metrics.items():
                result[key] = value
        return result

    def __eq__(self, other: "Individual") -> bool:
        if isinstance(other, Individual):
            # TODO: should call the Road __eq__ method
            return self.representation == other.representation
        raise RuntimeError("other {} is not an individual".format(type(other)))

    def __hash__(self) -> int:
        return self.representation.__hash__()
