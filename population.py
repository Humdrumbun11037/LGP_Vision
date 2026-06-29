"""Population management for Linear Genetic Programming."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from multiprocessing import Pool, cpu_count

import numpy as np


from individual import Individual  # type: ignore
from instruction_set import InstructionSet
from memory_system import MemoryConfig

from evaluator import FitnessEvaluator
from operators import GeneticOperators

# Module-level worker function for multiprocessing (must be at module level for pickle)
def _evaluate_worker(args: Tuple[int, Individual, type, dict]) -> Tuple[int, float]:
    """Worker function for parallel evaluation.
    
    Args:
        args: Tuple of (index, individual, EvaluatorClass, evaluator_kwargs)
        
    Returns:
        Tuple of (index, fitness)
    """
    idx, individual, EvaluatorClass, evaluator_kwargs = args
    
    # Use the same seed for all workers for deterministic evaluation
    
    base_seed = evaluator_kwargs.get('rng_seed', None)
    
    # Update the config with the base seed (same for all workers)
    evaluator_kwargs = evaluator_kwargs.copy()  # Don't modify original
    evaluator_kwargs['rng_seed'] = base_seed
    
    # Create a fresh evaluator (and thus fresh environment) for this worker
    # Check if we need to create a config object or use kwargs directly
    from evaluator import (
        BaseEvaluatorConfig, 
        CartPoleEvaluatorConfig, 
        AcrobotEvaluatorConfig,
        PendulumEvaluatorConfig,
        FlappyBirdEvaluatorConfig,
        FlappyBirdSimpleEvaluatorConfig
    )
    
    try:
        # Try to determine config type from EvaluatorClass
        if EvaluatorClass.__name__ == 'FlappyBirdEvaluator':
            # Create config object for FlappyBirdEvaluator with unique seed
            config = FlappyBirdEvaluatorConfig(**evaluator_kwargs)
            worker_evaluator = EvaluatorClass(config=config)
        elif EvaluatorClass.__name__ == 'FlappyBirdSimpleEvaluator':
            # Create config object for FlappyBirdSimpleEvaluator with unique seed
            config = FlappyBirdSimpleEvaluatorConfig(**evaluator_kwargs)
            worker_evaluator = EvaluatorClass(config=config)
        elif EvaluatorClass.__name__ == 'CartPoleEvaluator':
            # Create config object for CartPoleEvaluator with unique seed
            config = CartPoleEvaluatorConfig(**evaluator_kwargs)
            worker_evaluator = EvaluatorClass(config=config)
        elif EvaluatorClass.__name__ == 'AcrobotEvaluator':
            # Create config object for AcrobotEvaluator with unique seed
            config = AcrobotEvaluatorConfig(**evaluator_kwargs)
            worker_evaluator = EvaluatorClass(config=config)
        elif EvaluatorClass.__name__ == 'PendulumEvaluator':
            # Create config object for PendulumEvaluator with unique seed
            config = PendulumEvaluatorConfig(**evaluator_kwargs)
            worker_evaluator = EvaluatorClass(config=config)
        else:
            # For other evaluators, use kwargs directly
            evaluator_kwargs['rng'] = np.random.default_rng(base_seed)
            worker_evaluator = EvaluatorClass(**evaluator_kwargs)
    except Exception as e:
        # Fallback: try direct kwargs (backward compatibility)
        evaluator_kwargs['rng'] = np.random.default_rng(base_seed)
        worker_evaluator = EvaluatorClass(**evaluator_kwargs)
    
    try:
        # Evaluate the individual
        fitness = individual.evaluate(worker_evaluator)
        return idx, fitness
    except Exception as e:
        # Return a very poor fitness on error
        print(f"Warning: Evaluation failed for individual {idx}: {e}")
        return idx, float('-inf')
    finally:
        # Cleanup
        if hasattr(worker_evaluator, 'close'):
            worker_evaluator.close()


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
        individuals: List[Individual] = []
        for _ in range(self.config.size):
            ind = Individual.random(
                instruction_set=self.instruction_set,
                memory_config=self.memory_config,
                program_length=self._random_program_length(),
                rng=self.rng,
                mutate_constants=mutate_constants,
                max_program_length=self.config.max_program_length,
                restricted=True,   # ← only here: protect adaptive rate regs at init
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

    def _extract_evaluator_config(self, evaluator: 'FitnessEvaluator') -> dict:
        """Extract configuration from evaluator to recreate it in worker processes.
        
        Args:
            evaluator: The evaluator instance to extract config from
            
        Returns:
            Dictionary of parameters needed to recreate the evaluator
        """
        # Convert config dataclass to dict, but handle RNG seed properly
        config_dict = evaluator.config.__dict__.copy()
        # Don't pass output_registers if it's None (will be derived from output_register)
        if config_dict.get('output_registers') is None:
            config_dict.pop('output_registers', None)
        return config_dict

    def evaluate_all(
        self, 
        evaluator: 'FitnessEvaluator', 
        verbose: bool = False,
        n_jobs: Optional[int] = None
    ) -> None:
        """Evaluate all individuals in the population.
        
        Args:
            evaluator: Fitness evaluator to use
            verbose: Print progress updates
            n_jobs: Number of parallel workers. None = auto-detect (use all CPUs),
                    1 = sequential evaluation, >1 = use that many workers
        """
        if n_jobs == 1 or len(self.individuals) == 0:
            # Sequential fallback
            for idx, individual in enumerate(self.individuals):
                individual.evaluate(evaluator)
                if verbose and (idx + 1) % 10 == 0:
                    print(f"Evaluated {idx + 1}/{len(self.individuals)} individuals")
            return
        
        # Parallel evaluation
        if n_jobs is None:
            n_jobs = cpu_count()
        
        # Extract evaluator configuration
        evaluator_config = self._extract_evaluator_config(evaluator)
        evaluator_class = type(evaluator)
        
        # Prepare arguments for workers: (idx, individual, EvaluatorClass, config)
        args_list = [
            (idx, ind, evaluator_class, evaluator_config)
            for idx, ind in enumerate(self.individuals)
        ]
        
        # Evaluate in parallel
        try:
            with Pool(processes=n_jobs) as pool:
                results = pool.map(_evaluate_worker, args_list)
        except Exception as e:
            if verbose:
                print(f"Warning: Parallel evaluation failed: {e}")
                print("Falling back to sequential evaluation...")
            # Fallback to sequential
            for idx, individual in enumerate(self.individuals):
                individual.evaluate(evaluator)
                if verbose and (idx + 1) % 10 == 0:
                    print(f"Evaluated {idx + 1}/{len(self.individuals)} individuals")
            return
        
        # Update fitness values from results
        for idx, fitness in results:
            self.individuals[idx].fitness = fitness
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