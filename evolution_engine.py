"""Evolution loop for Linear Genetic Programming."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

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

    def run(self) -> Population:
        for gen in range(self.config.max_generations):
            restored_programs = self._prune_introns()
            self.population.evaluate_all(self.evaluator, verbose=self.config.verbose)
            for individual, original_program in restored_programs:
                individual.program = original_program

            self.population.record_statistics()
            
            # Track best agent of this generation
            self._record_best_agent(gen)

            if self.config.verbose:
                print(f"\n=== Generation {gen} ===")
                self.population.print_summary()

            offspring = []
            for _ in range(self.population.config.size):
                parent1, parent2 = self.population.tournament_selection(3, num_winners=2)
                child_program, _ = self.operators.crossover(parent1.program, parent2.program,self.config.mutation_threshold, self.rng)
                self.operators.mutate_program(
                    child_program,
                    self.config.mutation_threshold,
                    self.rng,
                    max_length=self.population.config.max_program_length,
                )
                if self.population.config.max_program_length is not None:
                    child_program.max_program_length = self.population.config.max_program_length

                child = parent1.create_offspring(parent_ids=(parent1.id, parent2.id))
                child.program = child_program
                if self.config.constant_mutation_rate > 0 and (
                    self.rng.random() < self.config.constant_mutation_rate
                ):
                    self.operators.mutate_constants(child.memory, self.rng)
                child.invalidate_fitness()
                offspring.append(child)

            offspring = self.population.apply_elitism(offspring)
            self.population.replace_population(offspring)

        if self.config.verbose:
            print("\nEvolution complete.")
        return self.population

    def _prune_introns(self) -> None:
        outputs = getattr(self.evaluator, "output_registers", None)
        if not outputs:
            return []
        restored = []
        for individual in self.population.individuals:
            pruned = individual.program.remove_introns(outputs)
            pruned.max_program_length = individual.program.max_program_length
            restored.append((individual, individual.program))
            individual.program = pruned
        return restored

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
            effective_length = best_agent.program.get_effective_length(output_registers)
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

    class DummyEvaluator(FitnessEvaluator):
        def _evaluate_episode(self, individual, episode_idx):
            return self.rng.normal()

    evaluator = DummyEvaluator(rng=rng)

    engine = EvolutionEngine(
        population=population,
        operators=operators,
        evaluator=evaluator,
        config=EvolutionConfig(max_generations=3, mutation_threshold=0.2, verbose=False),
        rng=rng,
    )
    engine.run()
    population.print_summary()
