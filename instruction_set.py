from memory_system import MemoryBank, MemoryType, MemoryConfig
from operation import Operation
from typing import List
from instruction import Instruction
import numpy as np
from program import Program
class InstructionSet:
    """
    Generates random valid instructions for typed LGP.
    
    Responsibilities:
    - Store available operations
    - Generate type-safe random instructions
    - Respect obs (read-only) vs working (read-write) distinction
    """
    
    def __init__(self, 
                 operations: List[Operation],
                memory_config: MemoryConfig):
        """
        Args:
            operations: List of operation instances to use
            n_scalar: Number of working scalar registers
            n_vector: Number of working vector registers
            n_matrix: Number of working matrix registers
            n_obs_scalar: Number of observation scalar registers
            n_obs_vector: Number of observation vector registers
            n_obs_matrix: Number of observation matrix registers
        """
        self.operations = operations
        self.n_scalar = memory_config.n_scalar
        self.n_vector = memory_config.n_vector
        self.n_matrix = memory_config.n_matrix
        self.n_obs_scalar = memory_config.n_obs_scalar
        self.n_obs_vector = memory_config.n_obs_vector
        self.n_obs_matrix = memory_config.n_obs_matrix

        
     
        # Pre-compute index ranges for efficiency
        self._source_ranges = {
            MemoryType.SCALAR: (
                list(range(-self.n_obs_scalar, 0)) +  # Obs: [-n, ..., -1]
                list(range(0, self.n_scalar))           # Work: [0, ..., n-1]
            ),
            MemoryType.VECTOR: (
                list(range(-self.n_obs_vector, 0)) +
                list(range(0, self.n_vector))
            ),
            MemoryType.MATRIX: (
                list(range(-self.n_obs_matrix, 0)) +
                list(range(0, self.n_matrix))
            ),
        }
        
        self._dest_ranges = {
            MemoryType.SCALAR: list(range(0, self.n_scalar)),    # Work only
            MemoryType.VECTOR: list(range(0, self.n_vector)),
            MemoryType.MATRIX: list(range(0, self.n_matrix)),
        }
    
    def generate_random_instruction(self, rng=None) -> Instruction:
        """
        Generate a random type-safe instruction.
        
        Args:
            rng: numpy random generator (optional)
        
        Returns:
            Valid Instruction
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Pick random operation
        op = rng.choice(self.operations)
        
        # Pick source registers based on operation's input types
        source_types = op.input_types()
        source_indices = []
        source_obs_flags = []
        
        for src_type in source_types:
            # Decide observation flag (50% probability)
            obs_flag = rng.random() < 0.5
            source_obs_flags.append(obs_flag)
            
            # Always generate large range index regardless of flag
            source_indices.append(self.get_random_source(src_type, rng))
        
        # Pick destination register (working registers only)
        dest_type = op.output_type()
        dest_index = rng.choice(self._dest_ranges[dest_type])
        
        return Instruction(op, dest_type, dest_index, source_types, source_indices, source_obs_flags)
    
    def generate_random_program(self, length: int, rng=None):
        """Generate a random program of given length"""
        instructions = [
            self.generate_random_instruction(rng) 
            for _ in range(length)
        ]
        return Program(instructions)  
    def get_random_operator(self, rng= None):
        if rng is None:
            rng = np.random.default_rng()
        return rng.choice(self.operations)
    def get_random_dest(self,dest_type,rng= None):
        if rng is None:
            rng = np.random.default_rng()
        dest_index = rng.choice(self._dest_ranges[dest_type])
        return dest_index
    def get_random_source(self, source_type, rng=None):
        """
        Generate a random source index in large range.
        
        All indices are generated in range [0, 10000) regardless of 
        observation flag. The flag determines how the index is interpreted
        during execution (modulo by matrix dimensions vs register counts).
        
        Args:
            source_type: Memory type (unused but kept for interface compatibility)
            rng: Optional random number generator
        
        Returns:
            Random integer in range [0, 10000)
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Always generate large range index - interpretation happens during execute()
        return int(rng.integers(0, 10000))



        

        

