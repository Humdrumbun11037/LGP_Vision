"""Population management for Linear Genetic Programming."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np


from individual import Individual  # type: ignore
from instruction_set import InstructionSet
from memory_system import MemoryConfig

from evaluator import FitnessEvaluator
from operators import GeneticOperators


@dataclass
class PopulationConfig:
    """Configuration for population initialization and management."""

    size: int
    program_length: Tuple[int, int]  # (min_len, max_len]
    elitism: int = 1
    max_program_length: Optional[int] = None

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("Population size must be positive")
        if self.elitism < 0:
            raise ValueError("Elitism must be non-negative")
        if self.elitism >= self.size:
            raise ValueError("Elitism must be smaller than population size")
        if isinstance(self.program_length, tuple):
            lo, hi = self.program_length
            if lo <= 0 or hi <= 0 or hi < lo:
                raise ValueError("Program length range must be positive with hi >= lo")
        else:
            raise TypeError("program_length must be a tuple of (min_len, max_len)")
        if self.max_program_length is not None and self.max_program_length <= 0:
            raise ValueError("max_program_length must be positive")


class Population:
    """Manages a collection of individuals for evolutionary computation."""

    def __init__(
        self,
        config: PopulationConfig,
        instruction_set: InstructionSet,
        memory_config: MemoryConfig,
        operators: Optional[GeneticOperators] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.config = config
        self.instruction_set = instruction_set
        self.memory_config = memory_config
        self.operators = operators
        self.rng = rng or np.random.default_rng()

        self.individuals: List[Individual] = []
        self.generation = 0
        self.best_ever: Optional[Individual] = None
        self.best_ever_generation: int = 0
        self.fitness_history: List[Tuple[float, float, float]] = []

    # ------------------------------------------------------------------
    # Initialization

    def _random_program_length(self) -> int:
        lo, hi = self.config.program_length
        max_len = self.config.max_program_length
        if max_len is not None:
            hi = min(hi, max_len)
            lo = min(lo, max_len)
        if hi < lo:
            hi = lo
        if lo == hi:
            return lo
        return int(self.rng.integers(lo, hi + 1))

    def initialize_random(self, mutate_constants: bool = True) -> None:
        """Populate with random individuals."""

        individuals: List[Individual] = []
        for _ in range(self.config.size):
            ind = Individual.random(
                instruction_set=self.instruction_set,
                memory_config=self.memory_config,
                program_length=self._random_program_length(),
                rng=self.rng,
                mutate_constants=mutate_constants,
                max_program_length=self.config.max_program_length,
            )
            if self.config.max_program_length is not None:
                ind.program.max_program_length = self.config.max_program_length
            individuals.append(ind)
        self.individuals = individuals
        self.generation = 0
        self.best_ever = None
        self.best_ever_generation = 0

    # ------------------------------------------------------------------
    # Selection

    def tournament_selection(
        self, tournament_size: int = 3, num_winners: int = 1
    ) -> List[Individual]:
        if tournament_size > len(self.individuals):
            raise ValueError("Tournament size larger than population")

        winners: List[Individual] = []
        for _ in range(num_winners):
            contenders = self.rng.choice(self.individuals, tournament_size, replace=False)
            best = max(contenders, key=lambda ind: ind.fitness or float("-inf"))
            winners.append(best)
        return winners

    def select_best(self, n: int = 1) -> List[Individual]:
        if n <= 0:
            return []
        return sorted(
            self.individuals,
            key=lambda ind: ind.fitness or float("-inf"),
            reverse=True,
        )[:n]

    # ------------------------------------------------------------------
    # Population management

    def replace_population(self, new_individuals: List[Individual]) -> None:
        if len(new_individuals) != self.config.size:
            raise ValueError("New population size mismatch")
        
        self.individuals = new_individuals
        if self.config.max_program_length is not None:
            for ind in self.individuals:
                ind.program.max_program_length = self.config.max_program_length
        self.generation += 1

 
    def get_elites(self) -> List[Individual]:
        """Get elite individuals from current population."""
        if self.config.elitism == 0:
            return []
        elites = self.select_best(self.config.elitism)
        # Optionally increment age if you want elites to age
        for elite in elites:
            elite.age += 1
        return elites


    # ------------------------------------------------------------------
    # Metrics & utilities

    def get_best(self) -> Individual:
        return max(self.individuals, key=lambda ind: ind.fitness or float("-inf"))

    def get_worst(self) -> Individual:
        return min(self.individuals, key=lambda ind: ind.fitness or float("inf"))

    def __len__(self) -> int:
        return len(self.individuals)

    def __getitem__(self, idx: int) -> Individual:
        return self.individuals[idx]

    def __iter__(self):
        return iter(self.individuals)

    def compute_statistics(self) -> Tuple[float, float, float, float]:
        fitnesses = [ind.fitness for ind in self.individuals if ind.fitness is not None]
        if not fitnesses:
            return 0.0, 0.0, 0.0, 0.0
        return (
            float(np.min(fitnesses)),
            float(np.mean(fitnesses)),
            float(np.max(fitnesses)),
            float(np.std(fitnesses)),
        )

    def record_statistics(self) -> None:
        stats = self.compute_statistics()
        self.fitness_history.append(stats[:3])

    def get_fitness_summary(self) -> str:
        mn, mean, mx, std = self.compute_statistics()
        return f"Min: {mn:.3f}, Mean: {mean:.3f}, Max: {mx:.3f}, Std: {std:.3f}"

    def get_diversity_metrics(self) -> Dict[str, float]:
        lengths = np.array([len(ind.program) for ind in self.individuals], dtype=float)
        return {
            "mean_length": float(np.mean(lengths)) if lengths.size else 0.0,
            "std_length": float(np.std(lengths)) if lengths.size else 0.0,
        }

    # ------------------------------------------------------------------
    # Evaluation

    def evaluate_all(self, evaluator: 'FitnessEvaluator', verbose: bool = False) -> None:
        
        for idx, individual in enumerate(self.individuals):
            individual.evaluate(evaluator)
            if verbose and (idx + 1) % 10 == 0:
                print(f"Evaluated {idx + 1}/{len(self.individuals)} individuals")

    def invalidate_all_fitness(self) -> None:
        for individual in self.individuals:
            individual.invalidate_fitness()

    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        print("=" * 60)
        print(f"Generation {self.generation} | Population size {len(self.individuals)}")
        print(self.get_fitness_summary())
        div = self.get_diversity_metrics()
        print(f"Length mean {div['mean_length']:.1f}, std {div['std_length']:.1f}")
        if self.best_ever is not None:
            print(
                f"Best ever fitness {self.best_ever.fitness:.3f} at generation {self.best_ever_generation}"
            )
        print("=" * 60)


if __name__ == "__main__":
    from operation import ALL_OPS
    from instruction_set import InstructionSet
    from memory_system import MemoryBank

    rng = np.random.default_rng(123)

    memory_cfg = MemoryConfig(
        n_scalar=6,
        n_vector=2,
        n_matrix=1,
        n_obs_scalar=2,
        n_obs_vector=1,
        n_obs_matrix=0,
        vector_size=4,
        matrix_shape=(3, 3),
    )

    template_memory = MemoryBank(
        n_scalar=memory_cfg.n_scalar,
        n_vector=memory_cfg.n_vector,
        n_matrix=memory_cfg.n_matrix,
        n_obs_scalar=memory_cfg.n_obs_scalar,
        n_obs_vector=memory_cfg.n_obs_vector,
        n_obs_matrix=memory_cfg.n_obs_matrix,
        vector_size=memory_cfg.vector_size,
        matrix_shape=memory_cfg.matrix_shape,
    )

    instr_set = InstructionSet([op() for op in ALL_OPS], template_memory)
    operators = GeneticOperators(instr_set, rng)

    config = PopulationConfig(size=8, program_length=(4, 8), elitism=2)
    population = Population(config, instr_set, memory_cfg, operators=operators, rng=rng)
    population.initialize_random(mutate_constants=True)

    print("Initialized population:")
    population.print_summary()

    winners = population.tournament_selection(tournament_size=3, num_winners=2)
    print("Tournament winner IDs:", [w.id for w in winners])

    # Clone as offspring (for demonstration only)
    offspring = [ind.copy(new_id=True) for ind in population.individuals]
    offspring = population.apply_elitism(offspring)
    population.replace_population(offspring)

    print("\nAfter replacement:")
    population.print_summary()