from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List, TYPE_CHECKING
import itertools
import numpy as np
from memory_system import MemoryBank, MemoryConfig, MemoryType
from program import Program
from instruction_set import InstructionSet
from operators import GeneticOperators

if TYPE_CHECKING:
    from evaluator import FitnessEvaluator

# Global counter for Individual IDs
_individual_counter = itertools.count()

@dataclass
class Individual:
    """A candidate solution with program and evolvable constants"""
    program: Program
    memory: MemoryBank
    id: int = field(default_factory=lambda: next(_individual_counter))
    fitness: Optional[float] = None
    age: int = 0
    parent_ids: Tuple[int, ...] = ()
    # Internal cache fields (not in __init__)
    _effective_program: Optional[Program] = field(default=None, init=False)
    _output_registers: Optional[List[Tuple[MemoryType, int]]] = field(default=None, init=False)

    def get_effective_program(self, output_registers: List[Tuple[MemoryType, int]]) -> Program:
        """Get intron-removed program, computing lazily if needed."""
        # Check if cache is valid
        if (self._effective_program is None or 
            self._output_registers != output_registers):
            # Cache invalid - recompute
            self._effective_program = self.program.copy()
            self._effective_program.intron_removal(output_registers)
            self._output_registers = output_registers
        return self._effective_program
    def evaluate(self, evaluator: 'FitnessEvaluator') -> float:
        """Evaluate fitness using given evaluator"""
        if self.fitness is None:
            self.fitness = evaluator.evaluate(self)
        return self.fitness
    
    def invalidate_fitness(self):
        """Mark fitness as needing re-evaluation"""
        self.fitness = None
        
        # Also clear effective program cache when program might have changed
        self._effective_program = None
        self._output_registers = None
    
    def copy(self, new_id: bool = True) -> 'Individual':
        """
        Deep copy of individual.
        
        Args:
            new_id: If True, assigns new ID (for offspring).
                    If False, keeps same ID (for cloning/caching).
        """
        program_copy = self.program.copy()
        program_copy.max_program_length = self.program.max_program_length
        return Individual(
            program=program_copy,
            memory=self.memory.copy(),
            id=next(_individual_counter) if new_id else self.id,
            fitness=self.fitness,
            age=self.age,
            parent_ids=self.parent_ids
        )
    
    def create_offspring(self, parent_ids: Tuple[int, ...]) -> 'Individual':
        """Create offspring with this individual's program but new ID and parents"""
        offspring = self.copy(new_id=True)
        offspring.parent_ids = parent_ids
        offspring.age = 0
        offspring.invalidate_fitness()
        return offspring
    
    def get_effective_length(self, output_registers: List[Tuple[MemoryType, int]]) -> int:
        """Get effective program length (for parsimony)."""
        effective_program = self.get_effective_program(output_registers)
        return len(effective_program)
    
    def get_constants(self) -> Dict:
        """Get evolvable constants (for analysis)"""
        return self.memory.get_constants()
    def get_intron_ratio(self, output_registers: List[Tuple[MemoryType, int]]) -> float:
        """Get the ratio of intron instructions to total instructions.
        
        Returns:
            Float in [0, 1] where 0 = no introns, 1 = all introns
        """
        total = len(self.program)
        if total == 0:
            return 0.0
        effective = self.get_effective_length(output_registers)
        return 1.0 - (effective / total)
    

    @classmethod
    def random(
        cls,
        instruction_set: InstructionSet,
        memory_config: MemoryConfig,
        program_length: int,
        rng: Optional[np.random.Generator] = None,
        mutate_constants: bool = True,
        max_program_length: Optional[int] = None,
    ) -> 'Individual':
        rng = rng or np.random.default_rng()
        program = instruction_set.generate_random_program(program_length, rng)
        if max_program_length is not None:
            program.max_program_length = max_program_length
        memory = MemoryBank(
            n_scalar=memory_config.n_scalar,
            n_vector=memory_config.n_vector,
            n_matrix=memory_config.n_matrix,
            n_obs_scalar=memory_config.n_obs_scalar,
            n_obs_vector=memory_config.n_obs_vector,
            n_obs_matrix=memory_config.n_obs_matrix,
            vector_size=memory_config.vector_size,
            matrix_shape=memory_config.matrix_shape,
            init_scalar_range=memory_config.init_scalar_range,
            init_vector_range=memory_config.init_vector_range,
            init_matrix_range=memory_config.init_matrix_range,
            rng=rng,  # Pass RNG for reproducible memory initialization
        )
        if mutate_constants:
            GeneticOperators(instruction_set, rng).mutate_constants(memory, rng)
        return cls(program=program, memory=memory)
    
    def __repr__(self) -> str:
        fitness_str = f"{self.fitness:.2f}" if self.fitness is not None else "None"
        return f"Individual(id={self.id}, fitness={fitness_str}, len={len(self.program)}, age={self.age})"


# Methods to move 
    # def get_effective_length(self, output_registers: List[Tuple[MemoryType, int]]) -> int:
    #     """
    #     Get the number of effective instructions.
        
    #     Useful for fitness metrics and bloat control.
    #     """
    #     return len(self.find_effective_instructions(output_registers))
    
    # def get_intron_ratio(self, output_registers: List[Tuple[MemoryType, int]]) -> float:
    #     """
    #     Get the ratio of intron instructions to total instructions.
        
    #     Returns:
    #         Float in [0, 1] where 0 = no introns, 1 = all introns
    #     """
    #     if len(self.instructions) == 0:
    #         return 0.0
    #     introns = self.get_introns(output_registers)
    #     return len(introns) / len(self.instructions)