"""Evolution loop for Linear Genetic Programming."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List
import pickle
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.special import expit  # sigmoid, maps any real → (0, 1)

from population import Population
from operators import GeneticOperators
from evaluator import FitnessEvaluator


# ---------------------------------------------------------------------------
# Number of adaptive mutation rate registers and their fixed offsets.
# Registers are placed immediately after the output register (index 0), so
# they occupy scalar indices 1, 2, 3, 4.
# ---------------------------------------------------------------------------
N_ADAPTIVE_RATE_REGISTERS = 4
ADAPTIVE_RATE_BASE_INDEX = 1   # first register index (output reg is 0)

# Semantic meaning of each adaptive rate register (for documentation / plots)
ADAPTIVE_RATE_NAMES = [
    "micro_mutation",       # register 1
    "add_instruction",      # register 2
    "delete_instruction",   # register 3
    "crossover_threshold",  # register 4
]


def _read_adaptive_rates(individual) -> List[float]:
    """
    Read the four adaptive mutation rate values from an individual's scalar
    registers and map them to [0, 1] via sigmoid.

    Returns a list of four floats in (0, 1).
    """
    rates = []
    for offset in range(N_ADAPTIVE_RATE_REGISTERS):
        idx = ADAPTIVE_RATE_BASE_INDEX + offset
        raw = float(individual.memory.read_scalar(idx))
        rates.append(float(expit(raw)))
    return rates


@dataclass
class EvolutionConfig:
    max_generations: int = 100
    mutation_threshold: float = 0.1
    constant_mutation_rate: float = 0.0
    crossover_threshold: float = 0.9
    verbose: bool = False
    # Checkpoint settings (managed by ExperimentManager)
    checkpoint_dir: Optional[str] = None  # Directory for checkpoints
    checkpoint_every: Optional[int] = None  # Save every N generations (None = only on improvement)
    # Stats logging (managed by ExperimentManager)
    stats_log_path: Optional[str] = None  # Full path to stats CSV file
    # Adaptive mutation rates
    adaptive_mutation_rates: bool = False  # Use per-individual evolved mutation rates

    def __post_init__(self) -> None:
        if self.max_generations <= 0:
            raise ValueError("max_generations must be > 0")
        if not (0.0 <= self.mutation_threshold <= 1.0):
            raise ValueError("mutation_threshold must be within [0, 1]")
        if not (0.0 <= self.constant_mutation_rate <= 1.0):
            raise ValueError("constant_mutation_rate must be within [0, 1]")


@dataclass
class BestAgentInfo:
    """Information about the best agent in a generation."""
    generation: int
    fitness: float
    effective_code_rate: float  # effective_length / total_length
    total_length: int
    effective_length: int
    # Adaptive mutation rates of the best agent (None when feature is disabled)
    adaptive_rates: Optional[List[float]] = None


@dataclass
class AdaptiveRateStats:
    """Per-generation population statistics for each adaptive mutation rate."""
    generation: int
    # Each field is a list of 4 values (one per rate), ordered by
    # ADAPTIVE_RATE_NAMES: [micro_mutation, add_instruction,
    #                        delete_instruction, crossover_threshold]
    mean: List[float] = field(default_factory=lambda: [0.0] * N_ADAPTIVE_RATE_REGISTERS)
    std: List[float] = field(default_factory=lambda: [0.0] * N_ADAPTIVE_RATE_REGISTERS)
    min: List[float] = field(default_factory=lambda: [0.0] * N_ADAPTIVE_RATE_REGISTERS)
    max: List[float] = field(default_factory=lambda: [0.0] * N_ADAPTIVE_RATE_REGISTERS)
    best: List[float] = field(default_factory=lambda: [0.0] * N_ADAPTIVE_RATE_REGISTERS)


class EvolutionEngine:
    """Coordinates evaluation, selection, and variation."""

    def __init__(
        self,
        population: Population,
        operators: GeneticOperators,
        evaluator: FitnessEvaluator,
        config: EvolutionConfig,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.population = population
        self.operators = operators
        self.evaluator = evaluator
        self.config = config
        self.rng = rng or np.random.default_rng()
        # Track best agent info per generation
        self.best_agent_history: List[BestAgentInfo] = []
        # Track adaptive rate population statistics per generation
        self.adaptive_rate_history: List[AdaptiveRateStats] = []
        # Track best fitness ever for checkpoint saving
        self.best_fitness_ever: Optional[float] = None
        # CSV statistics logging
        self._stats_file_path: Optional[Path] = None
        if self.config.stats_log_path is not None:
            self._init_stats_logging()

    # ------------------------------------------------------------------
    # Main evolution loop
    # ------------------------------------------------------------------

    def run(self, modes_tracker=None) -> Population:
        for gen in range(self.config.max_generations):
            # --- evaluate all ---
            n_jobs = self.evaluator.config.n_jobs
            self.population.evaluate_all(self.evaluator, verbose=self.config.verbose, n_jobs=n_jobs)

            # Track best-ever Individual (population is evaluated)
            current_best = self.population.get_best()
            
            if current_best.fitness is not None:
                was_none = self.population.best_ever is None
                if was_none or current_best.fitness > self.population.best_ever.fitness:
                    self.population.best_ever = current_best.copy(new_id=False)
                    self.population.best_ever_generation = self.population.generation
                    if self.config.verbose:
                        action = "Initialized" if was_none else "Updated"
                        print(f"  → {action} best_ever (fitness: {current_best.fitness:.4f}, gen: {gen})")
        
            # Record statistics and checkpoints
            self.population.record_statistics()
            self._record_best_agent(gen)
            if self.config.adaptive_mutation_rates:
                self._record_adaptive_rate_stats(gen)
            self._log_generation_stats(gen)
            self._check_and_save_checkpoint(gen)

            if modes_tracker is not None:
                modes_tracker.record(self.population, gen)
                modes_tracker.print_latest()

            if self.config.verbose:
                print(f"\n=== Generation {gen} ===")
                self.population.print_summary()
                if self.config.adaptive_mutation_rates and self.adaptive_rate_history:
                    stats = self.adaptive_rate_history[-1]
                    parts = [
                        f"{ADAPTIVE_RATE_NAMES[i]}={stats.mean[i]:.3f}±{stats.std[i]:.3f}"
                        for i in range(N_ADAPTIVE_RATE_REGISTERS)
                    ]
                    print(f"  Adaptive rates (pop mean±std): {', '.join(parts)}")

            if gen < self.config.max_generations - 1:
                # --- produce next generation ---
                elites = self.population.get_elites()
                offspring = elites
                
                while len(offspring) < self.population.config.size:
                    parent1, parent2 = self.population.tournament_selection(3, num_winners=2)

                    # Determine per-parent mutation / crossover rates
                    if self.config.adaptive_mutation_rates:
                        rates1 = _read_adaptive_rates(parent1)
                        rates2 = _read_adaptive_rates(parent2)
                        # Use average of the two parents for the shared crossover
                        # threshold so neither parent dominates.
                        micro_mut1, add_rate1, del_rate1, xover1 = rates1
                        micro_mut2, add_rate2, del_rate2, xover2 = rates2
                        crossover_threshold = (xover1 + xover2) / 2.0
                    else:
                        micro_mut1 = micro_mut2 = self.config.mutation_threshold
                        add_rate1 = add_rate2 = self.config.mutation_threshold
                        del_rate1 = del_rate2 = self.config.mutation_threshold
                        crossover_threshold = self.config.crossover_threshold

                    child_program_1, child_program_2 = self.operators.crossover(
                        parent1.program, parent2.program, crossover_threshold, self.rng
                    )

                    # Mutate child programs using (potentially adaptive) rates
                    if self.config.adaptive_mutation_rates:
                        self._mutate_program_adaptive(
                            child_program_1, micro_mut1, add_rate1, del_rate1,
                            self.population.config.max_program_length,
                        )
                        self._mutate_program_adaptive(
                            child_program_2, micro_mut2, add_rate2, del_rate2,
                            self.population.config.max_program_length,
                        )
                    else:
                        self.operators.mutate_program(
                            child_program_1,
                            self.config.mutation_threshold,
                            self.rng,
                            max_length=self.population.config.max_program_length,
                        )
                        self.operators.mutate_program(
                            child_program_2,
                            self.config.mutation_threshold,
                            self.rng,
                            max_length=self.population.config.max_program_length,
                        )
                
                    if self.population.config.max_program_length is not None:
                        child_program_1.max_program_length = self.population.config.max_program_length
                        child_program_2.max_program_length = self.population.config.max_program_length

                    # Create children inheriting parents' memory (and thus mutation rate
                    # registers), then mutate constants normally.
                    child1 = parent1.create_offspring(parent_ids=(parent1.id, parent2.id))
                    child1.program = child_program_1
                    
                    child2 = parent2.create_offspring(parent_ids=(parent1.id, parent2.id))
                    child2.program = child_program_2
                    
                    if self.config.constant_mutation_rate > 0 and (
                        self.rng.random() < self.config.constant_mutation_rate
                    ):
                        self.operators.mutate_constants(child1.memory, self.rng)
                        self.operators.mutate_constants(child2.memory, self.rng)
                        
                    child1.invalidate_fitness()
                    child2.invalidate_fitness()
                    offspring.append(child1)
                    offspring.append(child2)

                offspring = offspring[:self.population.config.size]
                self.population.replace_population(offspring)

        if self.config.verbose:
            print("\nEvolution complete.")
        return self.population

    # ------------------------------------------------------------------
    # Adaptive mutation helpers
    # ------------------------------------------------------------------

    def _mutate_program_adaptive(
        self,
        program,
        micro_mut_rate: float,
        add_rate: float,
        del_rate: float,
        max_length: Optional[int],
    ) -> None:
        """
        Walk every instruction and apply mutation operators using per-individual
        adaptive rates instead of the global config thresholds.

        Mutation probabilities per instruction:
          - micro_mut_rate  → apply micro-mutation (alter one field of the instruction)
          - add_rate        → insert a new random instruction after current position
          - del_rate        → delete the current instruction
        These are applied in separate passes to keep semantics clean.
        """
        rng = self.rng

        # --- micro-mutation pass ---
        for instr in list(program.instructions):
            if rng.random() < micro_mut_rate:
                self.operators.micro_mutate(instr, rng)

        # --- structural mutation pass (add / delete) ---
        i = 0
        while i < len(program.instructions):
            if rng.random() < add_rate:
                if max_length is None or len(program.instructions) < max_length:
                    self.operators.add_instruction_mutate(program, i, rng)
                    i += 1  # skip newly inserted instruction
            if i < len(program.instructions) and rng.random() < del_rate:
                self.operators.delete_instruction_mutate(program, i)
                i -= 1
            i += 1

        if len(program.instructions) == 0:
            program.instructions.append(
                self.operators.instruction_set.generate_random_instruction(rng)
            )

    # ------------------------------------------------------------------
    # Statistics recording
    # ------------------------------------------------------------------

    def _record_adaptive_rate_stats(self, generation: int) -> None:
        """Collect population-wide statistics for each adaptive rate register."""
        all_rates = np.array(
            [_read_adaptive_rates(ind) for ind in self.population.individuals],
            dtype=np.float64,
        )  # shape: (pop_size, 4)

        best_agent = self.population.get_best()
        best_rates = _read_adaptive_rates(best_agent) if best_agent is not None else [0.0] * 4

        stats = AdaptiveRateStats(
            generation=generation,
            mean=all_rates.mean(axis=0).tolist(),
            std=all_rates.std(axis=0).tolist(),
            min=all_rates.min(axis=0).tolist(),
            max=all_rates.max(axis=0).tolist(),
            best=best_rates,
        )
        self.adaptive_rate_history.append(stats)

    def _record_best_agent(self, generation: int) -> None:
        """Record information about the best agent in the current generation."""
        best_agent = self.population.get_best()
        
        if best_agent.fitness is None:
            return
        
        output_registers = getattr(self.evaluator, "output_registers", None)
        if output_registers is None:
            effective_length = len(best_agent.program)
            effective_code_rate = 1.0
        else:
            effective_length = best_agent.get_effective_length(output_registers)
            total_length = len(best_agent.program)
            effective_code_rate = effective_length / total_length if total_length > 0 else 0.0
        
        adaptive_rates = None
        if self.config.adaptive_mutation_rates:
            adaptive_rates = _read_adaptive_rates(best_agent)

        info = BestAgentInfo(
            generation=generation,
            fitness=best_agent.fitness,
            effective_code_rate=effective_code_rate,
            total_length=len(best_agent.program),
            effective_length=effective_length,
            adaptive_rates=adaptive_rates,
        )
        self.best_agent_history.append(info)
        
        if self.config.verbose:
            rate_str = ""
            if adaptive_rates is not None:
                rate_str = (
                    f" | adaptive_rates=["
                    + ", ".join(f"{r:.3f}" for r in adaptive_rates)
                    + "]"
                )
            print(f"Best agent: fitness={info.fitness:.4f}, "
                  f"effective_code_rate={info.effective_code_rate:.3f} "
                  f"({info.effective_length}/{info.total_length}){rate_str}")
    
    # ------------------------------------------------------------------
    # CSV statistics logging
    # ------------------------------------------------------------------

    def _init_stats_logging(self) -> None:
        """Initialize CSV statistics logging file with headers."""
        if self.config.stats_log_path is None:
            return
        
        self._stats_file_path = Path(self.config.stats_log_path)
        self._stats_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build adaptive rate column names
        adaptive_cols = []
        if self.config.adaptive_mutation_rates:
            for name in ADAPTIVE_RATE_NAMES:
                adaptive_cols += [
                    f"pop_mean_{name}",
                    f"pop_std_{name}",
                    f"best_{name}",
                ]

        try:
            with open(self._stats_file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'generation',
                    'best_fitness',
                    'mean_fitness',
                    'min_fitness',
                    'max_fitness',
                    'std_fitness',
                    'best_effective_code_rate',
                    'best_total_length',
                    'best_effective_length',
                    'mean_program_length',
                    'std_program_length',
                    'best_ever_fitness',
                    'best_ever_generation',
                ] + adaptive_cols)
            
            if self.config.verbose:
                print(f"Statistics logging: {self._stats_file_path}")
        except Exception as e:
            if self.config.verbose:
                print(f"Warning: Failed to initialize statistics logging: {e}")
            self._stats_file_path = None
    
    def _log_generation_stats(self, generation: int) -> None:
        """Log statistics for the current generation to CSV file."""
        if self._stats_file_path is None:
            return
        
        try:
            min_fitness, mean_fitness, max_fitness, std_fitness = self.population.compute_statistics()
            
            best_agent = self.population.get_best()
            if best_agent.fitness is None:
                return
            
            best_info = None
            if len(self.best_agent_history) > 0:
                for info in reversed(self.best_agent_history):
                    if info.generation == generation:
                        best_info = info
                        break
            
            if best_info is None:
                output_registers = getattr(self.evaluator, "output_registers", None)
                if output_registers is None:
                    effective_length = len(best_agent.program)
                    effective_code_rate = 1.0
                else:
                    effective_length = best_agent.get_effective_length(output_registers)
                    total_length = len(best_agent.program)
                    effective_code_rate = effective_length / total_length if total_length > 0 else 0.0
                best_total_length = len(best_agent.program)
            else:
                effective_code_rate = best_info.effective_code_rate
                best_total_length = best_info.total_length
                effective_length = best_info.effective_length
            
            diversity = self.population.get_diversity_metrics()
            mean_program_length = diversity['mean_length']
            std_program_length = diversity['std_length']
            
            best_ever_fitness = (
                self.population.best_ever.fitness 
                if self.population.best_ever is not None and self.population.best_ever.fitness is not None
                else None
            )
            best_ever_generation = (
                self.population.best_ever_generation 
                if self.population.best_ever is not None
                else None
            )

            # Adaptive rate columns
            adaptive_vals = []
            if self.config.adaptive_mutation_rates and self.adaptive_rate_history:
                rate_stats = self.adaptive_rate_history[-1]
                for i in range(N_ADAPTIVE_RATE_REGISTERS):
                    adaptive_vals += [
                        rate_stats.mean[i],
                        rate_stats.std[i],
                        rate_stats.best[i],
                    ]
            
            with open(self._stats_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    generation,
                    max_fitness,
                    mean_fitness,
                    min_fitness,
                    max_fitness,
                    std_fitness,
                    effective_code_rate,
                    best_total_length,
                    effective_length,
                    mean_program_length,
                    std_program_length,
                    best_ever_fitness if best_ever_fitness is not None else '',
                    best_ever_generation if best_ever_generation is not None else '',
                ] + adaptive_vals)
        except Exception as e:
            if self.config.verbose:
                print(f"Warning: Failed to log generation statistics: {e}")
    
    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _check_and_save_checkpoint(self, generation: int) -> None:
        """Check if checkpoint should be saved and save if needed."""
        if self.config.checkpoint_dir is None:
            return
        
        best_agent = self.population.get_best()
        if best_agent.fitness is None:
            return
        
        should_save = False
        
        if self.config.checkpoint_every is not None:
            if generation % self.config.checkpoint_every == 0:
                should_save = True
        
        is_improvement = (
            self.best_fitness_ever is None or 
            best_agent.fitness > self.best_fitness_ever
        )
        if is_improvement:
            self.best_fitness_ever = best_agent.fitness
            should_save = True
        
        if should_save:
            self._save_checkpoint(generation, best_agent.fitness)
    
    def _save_checkpoint(self, generation: int, fitness: float) -> None:
        """Save the population to a checkpoint file with generation number."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"gen_{generation:04d}.pkl"
        checkpoint_path = checkpoint_dir / filename
        
        checkpoint_data = {
            'generation': generation,
            'fitness': fitness,
            'population': self.population,
            'best_agent_history': self.best_agent_history,
            'adaptive_rate_history': self.adaptive_rate_history,
            'config': self.config,
        }
        
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            latest_path = checkpoint_dir / "latest.pkl"
            if latest_path.exists() or latest_path.is_symlink():
                latest_path.unlink()
            try:
                latest_path.symlink_to(filename)
            except OSError:
                import shutil
                shutil.copy2(checkpoint_path, latest_path)
            
            if self.config.verbose:
                print(f"Checkpoint saved: gen={generation}, fitness={fitness:.4f} -> {checkpoint_path}")
        except Exception as e:
            if self.config.verbose:
                print(f"Warning: Failed to save checkpoint: {e}")
    
    @staticmethod
    def load_checkpoint(checkpoint_path: str) -> dict:
        """Load a checkpoint file and return the saved data."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        return checkpoint_data


if __name__ == "__main__":
    from memory_system import MemoryConfig, MemoryBank
    from instruction_set import InstructionSet
    from operation import ALL_OPS

    rng = np.random.default_rng(0)

    memory_cfg = MemoryConfig(
        n_scalar=6,
        n_vector=2,
        n_matrix=1,
        n_obs_scalar=2,
        n_obs_vector=1,
        n_obs_matrix=1,
        vector_size=5,
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

    from population import PopulationConfig, Population

    pop_config = PopulationConfig(size=6, program_length=(3, 6), elitism=1)
    population = Population(pop_config, instr_set, memory_cfg, operators=operators, rng=rng)
    population.initialize_random(mutate_constants=True)

    from evaluator import BaseEvaluatorConfig
    
    class DummyEvaluator(FitnessEvaluator):
        def _evaluate_episode(self, individual, episode_idx):
            return self.rng.normal()

    dummy_config = BaseEvaluatorConfig(episodes=1, rng_seed=0)
    evaluator = DummyEvaluator(config=dummy_config)

    engine = EvolutionEngine(
        population=population,
        operators=operators,
        evaluator=evaluator,
        config=EvolutionConfig(max_generations=3, mutation_threshold=0.2, verbose=False),
        rng=rng,
    )
    engine.run()
    population.print_summary()