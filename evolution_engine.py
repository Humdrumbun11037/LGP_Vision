"""Evolution loop for Linear Genetic Programming."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List
import pickle
import csv
from pathlib import Path
from datetime import datetime

import numpy as np

from population import Population
from operators import GeneticOperators
from evaluator import FitnessEvaluator


@dataclass
class EvolutionConfig:
    max_generations: int = 100
    mutation_threshold: float = 0.1
    constant_mutation_rate: float = 0.0
    crossover_threshold: float = 0.9
    verbose: bool = True
    checkpoint_path: Optional[str] = None  # Path to save checkpoints (overwrites on improvement)
    stats_log_dir: Optional[str] = None  # Directory to save CSV statistics files

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
        # Track best fitness ever for checkpoint saving
        self.best_fitness_ever: Optional[float] = None
        # CSV statistics logging
        self._stats_file_path: Optional[Path] = None
        if self.config.stats_log_dir is not None:
            self._init_stats_logging()

    def run(self) -> Population:
        for gen in range(self.config.max_generations):
            # evaluate all 
            # Get n_jobs from evaluator config
            n_jobs = self.evaluator.config.n_jobs
            self.population.evaluate_all(self.evaluator, verbose=self.config.verbose, n_jobs=n_jobs)

            # Track best-ever Individual (population is evaluated)
            current_best = self.population.get_best()
            
            # Update best_ever if this is the first generation or if current is better
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
            self._log_generation_stats(gen)
            self._check_and_save_checkpoint(gen)



            if self.config.verbose:
                print(f"\n=== Generation {gen} ===")
                self.population.print_summary()
            if gen < self.config.max_generations - 1: # if last gen just evaluate dont replaces
                # add elites to new population 
                elites = self.population.get_elites()
                offspring = elites
                
                while len(offspring) < self.population.config.size:
                    parent1, parent2 = self.population.tournament_selection(3, num_winners=2)
                    child_program_1, child_program_2= self.operators.crossover(parent1.program, parent2.program,self.config.crossover_threshold, self.rng)
                    # mutate child 1 
                    self.operators.mutate_program(
                        child_program_1,
                        self.config.mutation_threshold,
                        self.rng,
                        max_length=self.population.config.max_program_length,
                    )
                    # mutate child 2
                    self.operators.mutate_program(
                        child_program_2,
                        self.config.mutation_threshold,
                        self.rng,
                        max_length=self.population.config.max_program_length,
                    )
                
                    if self.population.config.max_program_length is not None:
                        child_program_1.max_program_length = self.population.config.max_program_length
                        child_program_2.max_program_length = self.population.config.max_program_length

                    # Create child1 from parent1 (inherits parent1's memory/constants)
                    child1 = parent1.create_offspring(parent_ids=(parent1.id, parent2.id))
                    child1.program = child_program_1
                    
                    # Create child2 from parent2 (inherits parent2's memory/constants)
                    child2 = parent2.create_offspring(parent_ids=(parent1.id, parent2.id))
                    child2.program = child_program_2
                    # mutate memory of vchildren 
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

   

    def _record_best_agent(self, generation: int) -> None:
        """Record information about the best agent in the current generation."""
        best_agent = self.population.get_best()
        
        if best_agent.fitness is None:
            return  # Skip if not evaluated
        
        # Get output registers from evaluator
        output_registers = getattr(self.evaluator, "output_registers", None)
        if output_registers is None:
            # If no output registers specified, assume all code is effective
            effective_length = len(best_agent.program)
            effective_code_rate = 1.0
        else:
            # Calculate effective code metrics
            effective_length = best_agent.get_effective_length(output_registers)
            total_length = len(best_agent.program)
            effective_code_rate = effective_length / total_length if total_length > 0 else 0.0
        
        info = BestAgentInfo(
            generation=generation,
            fitness=best_agent.fitness,
            effective_code_rate=effective_code_rate,
            total_length=len(best_agent.program),
            effective_length=effective_length,
        )
        self.best_agent_history.append(info)
        
        if self.config.verbose:
            print(f"Best agent: fitness={info.fitness:.4f}, "
                  f"effective_code_rate={info.effective_code_rate:.3f} "
                  f"({info.effective_length}/{info.total_length})")
    
    def _init_stats_logging(self) -> None:
        """Initialize CSV statistics logging file with headers."""
        if self.config.stats_log_dir is None:
            return
        
        stats_dir = Path(self.config.stats_log_dir)
        stats_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"lgp_run_{timestamp}.csv"
        self._stats_file_path = stats_dir / filename
        
        # Write CSV header
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
                ])
            
            if self.config.verbose:
                print(f"✓ Statistics logging initialized: {self._stats_file_path}")
        except Exception as e:
            if self.config.verbose:
                print(f"⚠ Warning: Failed to initialize statistics logging: {e}")
            self._stats_file_path = None
    
    def _log_generation_stats(self, generation: int) -> None:
        """Log statistics for the current generation to CSV file."""
        if self._stats_file_path is None:
            return
        
        try:
            # Get fitness statistics from population
            min_fitness, mean_fitness, max_fitness, std_fitness = self.population.compute_statistics()
            
            # Get best agent info (from current generation's best)
            best_agent = self.population.get_best()
            if best_agent.fitness is None:
                # Skip logging if no valid fitness
                return
            
            # Get best agent details from best_agent_history (last entry is current generation)
            best_info = None
            if len(self.best_agent_history) > 0:
                # Find the entry for this generation
                for info in reversed(self.best_agent_history):
                    if info.generation == generation:
                        best_info = info
                        break
            
            # Fallback to computing if not in history
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
            
            # Get diversity metrics
            diversity = self.population.get_diversity_metrics()
            mean_program_length = diversity['mean_length']
            std_program_length = diversity['std_length']
            
            # Get best-ever info
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
            
            # Write row to CSV
            with open(self._stats_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    generation,
                    max_fitness,  # best_fitness (same as max_fitness)
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
                ])
        except Exception as e:
            if self.config.verbose:
                print(f"⚠ Warning: Failed to log generation statistics: {e}")
    
    def _check_and_save_checkpoint(self, generation: int) -> None:
        """Check if there's a fitness improvement and save checkpoint if enabled."""
        if self.config.checkpoint_path is None:
            return
        
        best_agent = self.population.get_best()
        if best_agent.fitness is None:
            return
        
        # Check if this is an improvement
        is_improvement = (
            self.best_fitness_ever is None or 
            best_agent.fitness > self.best_fitness_ever
        )
        
        if is_improvement:
            self.best_fitness_ever = best_agent.fitness
            self._save_checkpoint(generation, best_agent.fitness)
    
    def _save_checkpoint(self, generation: int, fitness: float) -> None:
        """Save the entire population to a checkpoint file."""
        checkpoint_path = Path(self.config.checkpoint_path)
        
        # Create directory if it doesn't exist
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare checkpoint data
        checkpoint_data = {
            'generation': generation,
            'fitness': fitness,
            'population': self.population,
            'best_agent_history': self.best_agent_history,
            'config': self.config,
        }
        
        # Save to file (overwrites previous checkpoint)
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            if self.config.verbose:
                print(f"✓ Checkpoint saved: generation={generation}, fitness={fitness:.4f} "
                      f"-> {checkpoint_path}")
        except Exception as e:
            if self.config.verbose:
                print(f"⚠ Warning: Failed to save checkpoint: {e}")
    
    @staticmethod
    def load_checkpoint(checkpoint_path: str) -> dict:
        """Load a checkpoint file and return the saved data.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            Dictionary containing:
                - 'generation': Generation number when saved
                - 'fitness': Best fitness when saved
                - 'population': The saved Population object
                - 'best_agent_history': History of best agents
                - 'config': EvolutionConfig used
                
        Example:
            >>> checkpoint = EvolutionEngine.load_checkpoint("checkpoints/best.pkl")
            >>> population = checkpoint['population']
            >>> generation = checkpoint['generation']
        """
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
