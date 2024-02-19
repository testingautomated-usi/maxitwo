import datetime
import itertools
import json
import logging
import os
import platform
import socket
import time
from typing import List, Tuple, Dict

import numpy as np

from code_pipeline.mock_evaluator import MockEvaluator
from code_pipeline.real_evaluator import RealEvaluator
from code_pipeline.visualization import RoadTestVisualizer
from config import (
    NUM_CONTROL_NODES,
    MAX_ANGLE,
    NUM_SAMPLED_POINTS,
    MOCK_SIM_NAME,
    MIN_ANGLE,
    BEAMNG_SIM_NAME,
    UDACITY_SIM_NAME,
    DONKEY_SIM_NAME,
)
from custom_types import GymEnv
from envs.beamng.config import MAP_SIZE
from factories import make_env, make_agent, make_test_generator
from global_log import GlobalLog
from self_driving.agent import Agent
from self_driving.road_utils import get_road
from self_driving.supervised_agent import SupervisedAgent
from test_generators.mapelites.config import (
    FEATURE_COMBINATIONS,
    CURVATURE_FEATURE_NAME,
    TURNS_COUNT_FEATURE_NAME,
)
from test_generators.mapelites.feature_axis import FeatureAxis
from test_generators.mapelites.individual import Individual
from test_generators.test_generator import TestGenerator
from utils.os_utils import (
    kill_beamng_simulator,
    kill_udacity_simulator,
    kill_donkey_simulator,
)
from utils.randomness import set_random_seed


# FIXME: refactor, it is not a test generator (or road generator I should say), put it in another folder.
#  Rename test_generators package with road_generator
from utils.report_utils import (
    write_mapelites_report,
    plot_map_of_elites,
    plot_raw_map_of_elites,
    write_individual_report,
    save_images_of_individuals,
)


class MapElites:

    def __init__(
        self,
        env: GymEnv,
        env_name: str,
        agent: Agent,
        filepath: str,
        min_angle: int,
        max_angle: int,
        population_size: int,
        mutation_extent: int,
        iteration_runtime: int,
        test_generator: TestGenerator,
        mock_evaluator: bool = False,
        feature_combination: str = "{}-{}".format(
            TURNS_COUNT_FEATURE_NAME, CURVATURE_FEATURE_NAME
        ),
        merged_heatmap: bool = False,
        individual_migration: bool = False,
        seed: int = 0,
        port: int = -1,
        donkey_exe_path: str = None,
        udacity_exe_path: str = None,
        beamng_home: str = None,
        beamng_user: str = None,
        headless: bool = False,
        beamng_autopilot: bool = False,
        restart_beamng_every: int = 10,
        restart_beamng_after_population: bool = True,
        run_id: int = -1,
        str_datetime: str = None,
        collect_images: bool = False,
        cyclegan_experiment_name: str = None,
        gpu_ids: str = "-1",
        cyclegan_checkpoints_dir: str = None,
        cyclegan_epoch: int = None,
    ):

        assert filepath is not None, "Filepath must be a string"
        assert os.path.exists(filepath), "Filepath {} does not exist".format(filepath)

        if not merged_heatmap and not individual_migration:
            if run_id != -1 and str_datetime is not None:
                self.filepath = os.path.join(
                    filepath,
                    "mapelites_{}_{}_{}".format(env_name, str_datetime, run_id),
                )
            else:
                self.filepath = os.path.join(
                    filepath,
                    "mapelites_{}_{}".format(
                        env_name, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                    ),
                )
        elif merged_heatmap:
            self.filepath = os.path.join(
                filepath,
                "mapelites_merged_{}_{}".format(
                    env_name, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                ),
            )
        elif individual_migration:
            self.filepath = os.path.join(
                filepath,
                "mapelites_migration_{}_{}".format(
                    env_name, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                ),
            )

        os.makedirs(name=self.filepath, exist_ok=False)

        logging.basicConfig(
            filename=os.path.join(self.filepath, "log.txt"),
            filemode="w",
            level=logging.DEBUG,
        )

        self.env = env
        self.env_name = env_name
        self.agent = agent
        self.min_angle = min_angle
        self.max_angle = max_angle
        assert (
            self.max_angle > self.min_angle
        ), "Max angle of road must be > min angle {}".format(self.min_angle)
        self.population_size = population_size
        self.mutation_extent = mutation_extent
        self.feature_combination = feature_combination
        self.iteration_runtime = iteration_runtime

        assert (
            self.feature_combination in FEATURE_COMBINATIONS
        ), "Feature combination {} not supported".format(feature_combination)

        self.logg = GlobalLog("MapElites")
        self.mock_evaluator = mock_evaluator
        self.collect_images = collect_images
        self.initialize_evaluator()

        self.test_generator = test_generator
        self.population: Dict[Tuple[int, int], Individual] = dict()
        self.population_with_all_individuals: Dict[
            Tuple[int, int], List[Individual]
        ] = dict()
        self.fitness_values: Dict[Tuple[int, int], float] = dict()
        self.all_individuals: List[Individual] = []
        self.feature_map = self.generate_feature_map(
            feature_combination=feature_combination
        )

        self.seed = seed
        self.port = port
        self.donkey_exe_path = donkey_exe_path
        self.udacity_exe_path = udacity_exe_path
        self.beamng_home_path = beamng_home
        self.beamng_user_path = beamng_user
        self.headless = headless
        self.beamng_autopilot = beamng_autopilot
        self.restart_beamng_every = restart_beamng_every
        self.restart_beamng_after_population = restart_beamng_after_population

        self.cyclegan_experiment_name = cyclegan_experiment_name
        self.gpu_ids = gpu_ids
        self.cyclegan_checkpoints_dir = cyclegan_checkpoints_dir
        self.cyclegan_epoch = cyclegan_epoch

        self.save_params()

    def initialize_evaluator(self) -> None:
        if not self.mock_evaluator:
            self.evaluator = RealEvaluator(
                env_name=self.env_name,
                env=self.env,
                agent=self.agent,
                feature_combination=self.feature_combination,
                collect_images=self.collect_images,
            )
        else:
            self.evaluator = MockEvaluator(feature_combination=self.feature_combination)

    def save_params(self) -> None:
        filename = os.path.join(self.filepath, "params.json")
        report = dict()

        report["env_name"] = self.env_name
        report["agent"] = str(type(self.agent))
        if isinstance(self.agent, SupervisedAgent):
            report["model_path"] = self.agent.model_path
        report["test_generator"] = str(type(self.test_generator))
        report["min_angle"] = self.min_angle
        report["max_angle"] = self.max_angle
        report["population_size"] = self.population_size
        report["mutation_extent"] = self.mutation_extent
        report["feature_combination"] = self.feature_combination
        report["iteration_runtime"] = self.iteration_runtime

        with open(filename, "w") as f:
            json.dump(report, f, sort_keys=False, indent=4)

    @staticmethod
    def generate_feature_map(feature_combination: str) -> List[FeatureAxis]:
        assert (
            "-" in feature_combination
        ), "The '-' symbol must be present in the feature_combination string"
        feature_names = feature_combination.split("-")
        return [
            FeatureAxis(bins=1, name=feature_name) for feature_name in feature_names
        ]

    def random_selection(self) -> Tuple[int, int]:
        """
        Select an elite x from the current map of elites.
        The selection is done by selecting a random bin for each feature
        dimension, until a bin with a value is found.
        """
        idx = np.random.choice(
            a=np.arange(start=0, stop=len(self.population.keys())), size=1
        )[0]
        return list(self.population.keys())[idx]

    def run(self, last: bool = False):

        # 1. generate roads with angles binned
        # 2. place roads in map to have initial population -> for each road, execute sim, tuple of features,
        # coordinates in the map (int casting)
        # 2.1 if there are two individuals in the same place local competition with fitness value
        # 3. until budget take one with ranked selection or random, mutate, execute and compute features and
        # coordinates in the map. If cell occupied local competition else expand map and place it

        start_time = time.perf_counter()
        # start by creating an initial set of random solutions
        self.generate_initial_population(start_time=start_time)
        if self.restart_beamng_after_population and self.env_name == BEAMNG_SIM_NAME:
            # Indeed the reset runtime slows down with the number of resets
            self.logg.info(
                "Restarting BeamNG to speed up the simulations. Restart after generating initial population."
            )
            self.restart_simulator(env_name=self.env_name)

        # iteration counter
        iteration = 0

        while iteration <= self.iteration_runtime:
            self.logg.info(
                "Time elapsed: {:.2f}s".format(time.perf_counter() - start_time)
            )
            feature_bin = self.random_selection()

            self.population[feature_bin].selected_counter += 1
            parent = self.population[feature_bin]
            self.logg.info(
                f"Iteration {iteration}: Selecting individual {parent.id} at {feature_bin}"
            )

            self.logg.debug(
                "Before mutation: {}".format(
                    [(cp.x, cp.y, cp.z) for cp in parent.representation.control_points]
                )
            )
            # mutate the individual
            mutated_individual = parent.mutate(mutation_extent=self.mutation_extent)
            self.logg.debug(
                "After mutation: {}".format(
                    [
                        (cp.x, cp.y, cp.z)
                        for cp in mutated_individual.representation.control_points
                    ]
                )
            )

            self.execute_individual(individual=mutated_individual)

            self.all_individuals.append(mutated_individual)

            # place the new individual in the map of elites
            self.place_in_mapelites(
                individual=mutated_individual,
                parent_feature_bin=feature_bin,
                iteration=iteration,
            )

            if (
                iteration != 0
                and iteration % self.restart_beamng_every == 0
                and self.env_name == BEAMNG_SIM_NAME
            ):
                # Indeed the reset runtime slows down with the number of resets
                self.logg.info(
                    "Restarting BeamNG to speed up the simulations. Iterations: {}".format(
                        iteration
                    )
                )
                self.restart_simulator(env_name=self.env_name)

            iteration += 1

        self.logg.info(
            "Runtime expired: {} iterations > {} iterations".format(
                iteration, self.iteration_runtime
            )
        )
        self.extract_results(iteration_count=iteration)
        if self.env_name == BEAMNG_SIM_NAME:
            self.evaluator.close()
        elif last and (
            self.env_name == UDACITY_SIM_NAME or self.env_name == DONKEY_SIM_NAME
        ):
            self.evaluator.close()

    def restart_simulator(
        self, env_name: str, kill_simulator_process: bool = False
    ) -> None:
        if kill_simulator_process:
            if env_name == BEAMNG_SIM_NAME:
                kill_beamng_simulator()
            elif env_name == UDACITY_SIM_NAME:
                kill_udacity_simulator()
            elif env_name == DONKEY_SIM_NAME:
                kill_donkey_simulator()
            else:
                raise NotImplementedError(f"Unknown simulator: {env_name}")
        else:
            self.evaluator.close()
        time.sleep(5)
        self.env = make_env(
            simulator_name=self.env_name,
            seed=self.seed,
            port=self.port,
            test_generator=self.test_generator,
            donkey_exe_path=self.donkey_exe_path,
            udacity_exe_path=self.udacity_exe_path,
            beamng_home=self.beamng_home_path,
            beamng_user=self.beamng_user_path,
            headless=self.headless,
            beamng_autopilot=self.beamng_autopilot,
            cyclegan_experiment_name=self.cyclegan_experiment_name,
            gpu_ids=self.gpu_ids,
            cyclegan_checkpoints_dir=self.cyclegan_checkpoints_dir,
            cyclegan_epoch=self.cyclegan_epoch,
        )
        self.initialize_evaluator()

    def execute_individual(self, individual: Individual) -> None:
        exception_thrown = True
        time_elapsed_exception = 0.0
        while exception_thrown:
            start_time_execution = time.perf_counter()
            try:

                self.evaluator.run_sim(individual=individual)
                exception_thrown = False

            except RuntimeError as e:
                self.logg.warn(
                    f"RuntimeError {e}: closing and restarting the simulator"
                )
                time_elapsed_exception += time.perf_counter() - start_time_execution
                if self.env_name != UDACITY_SIM_NAME:
                    exception_thrown = True
                    self.restart_simulator(env_name=self.env_name)
                self.logg.warn(
                    "Time elapsed during exception: {:.2f}".format(
                        time_elapsed_exception
                    )
                )

            except socket.timeout:
                self.logg.warn("socket error: killing and restarting the simulator")
                time_elapsed_exception += time.perf_counter() - start_time_execution
                exception_thrown = True
                assert (
                    self.env_name == BEAMNG_SIM_NAME
                ), "Sim name {} not supported during socket timeout".format(
                    self.env_name
                )
                self.restart_simulator(
                    env_name=self.env_name, kill_simulator_process=True
                )
                self.logg.warn(
                    "Time elapsed during exception: {:.2f}".format(
                        time_elapsed_exception
                    )
                )

            except TypeError:
                # TypeError: cannot unpack non-iterable NoneType object
                self.logg.warn("type error: killing and restarting the simulator")
                time_elapsed_exception += time.perf_counter() - start_time_execution
                exception_thrown = True
                assert (
                    self.env_name == BEAMNG_SIM_NAME
                ), "Sim name {} not supported during socket timeout".format(
                    self.env_name
                )
                self.restart_simulator(
                    env_name=self.env_name, kill_simulator_process=True
                )
                self.logg.warn(
                    "Time elapsed during exception: {:.2f}".format(
                        time_elapsed_exception
                    )
                )

            except Exception as ex:
                self.logg.warn(f"Generic exception: {ex}")
                time_elapsed_exception += time.perf_counter() - start_time_execution
                if self.env_name != UDACITY_SIM_NAME:
                    exception_thrown = True
                    self.restart_simulator(env_name=self.env_name)
                self.logg.warn(
                    "Time elapsed during exception: {:.2f}".format(
                        time_elapsed_exception
                    )
                )

    def execute_individuals_and_place_in_map(
        self, individuals: List[Individual], occupation_map: bool = True
    ) -> None:

        max_id = max([individual.id for individual in individuals])

        for i, individual in enumerate(individuals):
            self.logg.info("Executing individual {}/{}".format(i + 1, len(individuals)))

            road = get_road(
                road_points=individual.get_representation().road_points,
                control_points=individual.get_representation().control_points,
                road_width=individual.get_representation().road_width,
                simulator_name=self.env_name,
            )

            new_individual = Individual(road=road, start_id=max_id)
            new_individual.id = individual.id

            if len(individual.get_representation().road_points) != len(
                new_individual.get_representation().road_points
            ):
                road_test_visualizer = RoadTestVisualizer(map_size=MAP_SIZE)
                road_test_visualizer.visualize_road_test(
                    road=individual.get_representation(),
                    folder_path="logs",
                    filename="road_old",
                )
                road_test_visualizer.visualize_road_test(
                    road=new_individual.get_representation(),
                    folder_path="logs",
                    filename="road_new",
                )

            assert len(individual.get_representation().road_points) == len(
                new_individual.get_representation().road_points
            ), "Road point of old individual {} must be equal to road points of new individual {}".format(
                len(individual.get_representation().road_points),
                len(new_individual.get_representation().road_points),
            )

            self.all_individuals.append(new_individual)
            self.execute_individual(individual=new_individual)
            self.place_in_mapelites(
                individual=new_individual, iteration=0, occupation_map=occupation_map
            )

            if (
                i != 0
                and i % self.restart_beamng_every == 0
                and self.env_name == BEAMNG_SIM_NAME
            ):
                # Indeed the reset runtime slows down with the number of resets
                self.logg.info(
                    "Restarting BeamNG to speed up the simulations. Iterations: {}".format(
                        i
                    )
                )
                self.restart_simulator(env_name=self.env_name)

        self.extract_results(iteration_count=0, occupation_map=occupation_map)

        self.evaluator.close()

    def generate_initial_population(self, start_time: float):
        self.logg.info("Generate initial population")
        angles = np.arange(start=self.min_angle, stop=self.max_angle + 1, step=10)
        weights = np.ones(
            shape=(
                len(
                    angles,
                )
            )
        )
        probs = np.exp(weights - np.max(weights))  # to avoid numerical problems
        softmax_values = probs / np.sum(probs)

        for i in range(self.population_size):
            self.logg.info(
                "Time elapsed: {:.2f}s".format(time.perf_counter() - start_time)
            )
            random_angle = np.random.choice(a=angles, size=1, p=softmax_values)[0]
            idx = np.where(angles == random_angle)[0]
            # making angle with index idx less probable at the next iteration
            weights[idx] = np.clip(a=weights[idx] - 0.2, a_min=0.01, a_max=1.0)
            probs = np.exp(weights - np.max(weights))  # to avoid numerical problem
            softmax_values = probs / np.sum(probs)

            self.logg.debug("Creating road with max_angle: {}".format(random_angle))
            self.test_generator.set_max_angle(max_angle=random_angle)
            road = self.test_generator.generate()
            individual = Individual(road=road)

            self.execute_individual(individual=individual)
            self.all_individuals.append(individual)

            self.place_in_mapelites(individual=individual, iteration=0)

        self.extract_results(iteration_count=0)

    def map_individual_to_bin_in_map(self, individual: Individual) -> Tuple[int, int]:
        feature_bin = []
        for i, feature_axis in enumerate(self.feature_map):
            feature = individual.get_features()[i]
            if feature.get_value() < feature_axis.min:
                feature_axis.min = feature.get_value()
            feature_bin.append(feature.get_value())
        return tuple(feature_bin)

    def place_in_mapelites(
        self,
        individual: Individual,
        iteration: int,
        parent_feature_bin: Tuple[int, int] = None,
        occupation_map: bool = True,
    ) -> None:
        """
        Puts a solution inside the N-dimensional map of elites space.
        The following criteria is used:
        - Compute the feature descriptor of the solution to find the correct
                cell in the N-dimensional space
        - Compute the performance of the solution
        - Check if the cell is empty or if the previous performance is worse
            - Place new solution in the cell
        """
        fitness_value = individual.get_fitness().get_value()
        # get coordinates in the feature space
        feature_bin = self.map_individual_to_bin_in_map(individual=individual)

        for i in range(len(feature_bin)):
            # if the bin is not already present in the map
            if feature_bin[i] >= self.feature_map[i].bins:
                self.feature_map[i].bins = feature_bin[i] + 1

        if occupation_map:
            # compares value of x with value of the individual already in the bin
            if feature_bin in self.fitness_values:
                if fitness_value < self.fitness_values[feature_bin]:
                    self.logg.info(
                        f"Iteration {iteration}: Replacing individual {individual.id} "
                        f"at {feature_bin} with fitness {fitness_value}"
                    )

                    if parent_feature_bin is not None:
                        self.population[parent_feature_bin].placed_mutant += 1

                    self.fitness_values[feature_bin] = fitness_value
                    self.population[feature_bin] = individual
                else:
                    self.logg.info(
                        f"Iteration {iteration}: Rejecting individual {individual.id} at {feature_bin} with fitness "
                        f"{fitness_value} in favor of {self.fitness_values[feature_bin]}"
                    )
            else:
                self.logg.info(
                    f"Iteration {iteration}: Placing individual {individual.id} at {feature_bin} "
                    f"with fitness {fitness_value}"
                )
                if parent_feature_bin is not None:
                    self.population[parent_feature_bin].placed_mutant += 1
                self.fitness_values[feature_bin] = fitness_value
                self.population[feature_bin] = individual
        else:
            self.logg.info(
                f"Iteration {iteration}: Placing individual {individual.id} at {feature_bin} "
                f"with success flag {fitness_value > 0.0} and fitness {fitness_value}"
            )
            if feature_bin not in self.population_with_all_individuals:
                self.population_with_all_individuals[feature_bin] = []
            self.population_with_all_individuals[feature_bin].append(individual)

    def extract_results(
        self, iteration_count: int, occupation_map: bool = True
    ) -> None:

        feature_axes_names = [feature_axis.name for feature_axis in self.feature_map]

        # TODO: maybe the following is done to support more than features in the future
        for feature_axis_1, feature_axis_2 in itertools.combinations(
            self.feature_map, 2
        ):

            x = feature_axes_names.index(feature_axis_1.name)
            y = feature_axes_names.index(feature_axis_2.name)

            # Define a new 2-D dict
            population = dict()

            if occupation_map:

                # Define a new 2-D dict
                fitness_values = dict()

                for feature_bin, fitness_value in self.fitness_values.items():
                    feature_bin_2d = (feature_bin[x], feature_bin[y])
                    if feature_bin_2d in fitness_values:
                        if fitness_values[feature_bin_2d] > fitness_value:
                            fitness_values[feature_bin_2d] = fitness_value
                            population[feature_bin_2d] = self.population[feature_bin]
                    else:
                        fitness_values[feature_bin_2d] = fitness_value
                        population[feature_bin_2d] = self.population[feature_bin]

                self.logg.info("Plot map of elites end iterations")

                write_mapelites_report(
                    filepath=self.filepath,
                    iterations=iteration_count,
                    population=self.population,
                    fitness_values=fitness_values.values(),
                    individuals=self.all_individuals,
                )

                plot_map_of_elites(
                    data=fitness_values,
                    filepath=self.filepath,
                    iterations=iteration_count,
                    x_axis_label=feature_axis_1.name,
                    y_axis_label=feature_axis_2.name,
                    min_value_cbar=self.all_individuals[0]
                    .get_fitness()
                    .get_min_value(),
                    max_value_cbar=self.all_individuals[0]
                    .get_fitness()
                    .get_max_value(),
                )

                plot_raw_map_of_elites(
                    data=fitness_values,
                    filepath=self.filepath,
                    iterations=iteration_count,
                    x_axis_label=feature_axis_1.name,
                    y_axis_label=feature_axis_2.name,
                )

                if self.collect_images:
                    save_images_of_individuals(
                        filepath=self.filepath, population=self.population
                    )

            else:

                for (
                    feature_bin,
                    individuals,
                ) in self.population_with_all_individuals.items():
                    feature_bin_2d = (feature_bin[x], feature_bin[y])
                    fitness_values_individuals = [
                        individual.get_fitness().get_value()
                        for individual in self.population_with_all_individuals[
                            feature_bin
                        ]
                    ]
                    is_success_flags = [
                        fitness_value > 0.0
                        for fitness_value in fitness_values_individuals
                    ]
                    population[feature_bin_2d] = np.mean(is_success_flags)
                    self.logg.info(
                        "Feature bin: {}, Fitness values: {}, Mean probability: {}".format(
                            feature_bin,
                            fitness_values_individuals,
                            np.mean(is_success_flags),
                        )
                    )

                all_individuals = [
                    individual
                    for individuals in self.population_with_all_individuals.values()
                    for individual in individuals
                ]
                fitness_values = [
                    individual.get_fitness().get_value()
                    for individual in all_individuals
                ]

                write_mapelites_report(
                    filepath=self.filepath,
                    iterations=iteration_count,
                    population=None,
                    fitness_values=fitness_values,
                    individuals=all_individuals,
                )

                plot_raw_map_of_elites(
                    data=population,
                    filepath=self.filepath,
                    iterations=iteration_count,
                    x_axis_label=feature_axis_1.name,
                    y_axis_label=feature_axis_2.name,
                    occupation_map=False,
                )

                write_individual_report(
                    filepath=self.filepath,
                    population=self.population_with_all_individuals,
                )

                plot_map_of_elites(
                    data=population,
                    filepath=self.filepath,
                    iterations=0,
                    x_axis_label=feature_axis_1.name,
                    y_axis_label=feature_axis_2.name,
                    min_value_cbar=0.0,
                    max_value_cbar=1.0,
                    occupation_map=False,
                )

                if self.collect_images:
                    save_images_of_individuals(
                        filepath=self.filepath,
                        population=self.population_with_all_individuals,
                    )


if __name__ == "__main__":

    # env_name = DONKEY_SIM_NAME
    env_name = MOCK_SIM_NAME
    seed = 0
    platform_ = platform.system()
    agent_type = "supervised"
    generator_name = "random"
    donkey_exe_path = "../../../../Downloads/DonkeySimMacRepl/donkey_sim.app"
    udacity_exe_path = "../../../../Downloads/UdacitySimMacRepl/udacity_sim.app"
    population_size = 10
    mutation_extent = 6
    runtime = 30
    min_angle = MIN_ANGLE
    max_angle = MAX_ANGLE
    num_control_nodes = NUM_CONTROL_NODES
    num_spline_nodes = NUM_SAMPLED_POINTS

    set_random_seed(seed=seed)

    assert platform_.lower() == "darwin", "Only on MacOS for now"
    assert os.path.exists(donkey_exe_path), "Donkey executor file not found: {}".format(
        donkey_exe_path
    )
    assert os.path.exists(
        udacity_exe_path
    ), "Udacity executor file not found: {}".format(udacity_exe_path)

    env = make_env(
        simulator_name=env_name, seed=seed, donkey_exe_path=donkey_exe_path, port=-1
    )

    model_path = "../../logs/models/mixed-dave2-2022_06_04_14_03_27.h5"  # robust model
    # model_path = '../../logs/models/mixed-dave2-2022_06_07_15_51_20.h5'  # weak model
    assert os.path.exists(model_path), "Model path not found: {}".format(model_path)

    agent = make_agent(
        env_name=env_name,
        env=env,
        model_path=model_path,
        agent_type=agent_type,
        predict_throttle=False,
    )

    test_generator = make_test_generator(
        generator_name=generator_name,
        map_size=MAP_SIZE,
        simulator_name=env_name,
        agent_type=agent_type,
        num_control_nodes=num_control_nodes,
        max_angle=max_angle,
        num_spline_nodes=num_spline_nodes,
    )

    mapelites = MapElites(
        env=env,
        env_name=env_name,
        agent=agent,
        filepath="../../logs",
        min_angle=min_angle,
        max_angle=max_angle,
        mutation_extent=mutation_extent,
        population_size=population_size,
        mock_evaluator=env_name == MOCK_SIM_NAME,
        iteration_runtime=runtime,
        test_generator=test_generator,
    )
    mapelites.run()
