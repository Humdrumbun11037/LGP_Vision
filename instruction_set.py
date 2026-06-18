from memory_system import MemoryBank, MemoryType, MemoryConfig
from operation import Operation
from typing import List, Optional
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
    - Optionally protect adaptive mutation rate registers from being written
      during random initialisation (registers 1-4, i.e. just after the output
      register at index 0).
    """
    
    def __init__(self, 
                 operations: List[Operation],
                 memory_config: MemoryConfig,
                 protected_scalar_dest_indices: Optional[List[int]] = None):
        """
        Args:
            operations: List of operation instances to use
            memory_config: Memory configuration with register counts and dimensions
            protected_scalar_dest_indices: Scalar destination register indices that
                random instruction generation must never write to.  Used to protect
                the adaptive mutation rate registers so that they start as pure
                evolvable constants and are shaped only by selection.
        """
        self.operations = operations
        self.memory_config = memory_config  # Store for later use
        self.n_scalar = memory_config.n_scalar
        self.n_vector = memory_config.n_vector
        self.n_matrix = memory_config.n_matrix
        self.n_obs_scalar = memory_config.n_obs_scalar
        self.n_obs_vector = memory_config.n_obs_vector
        self.n_obs_matrix = memory_config.n_obs_matrix
        self.vector_size = memory_config.vector_size
        self.matrix_shape = memory_config.matrix_shape

        # Indices of scalar destination registers that are off-limits for
        # randomly generated instructions.  These registers act as evolvable
        # constants for adaptive mutation rates.
        self._protected_scalar_dests: List[int] = list(protected_scalar_dest_indices or [])

        # Pre-compute available observation types (those with n > 0)
        self._available_obs_types = self._get_available_obs_types()
        
        # Pre-compute index ranges for efficiency.
        # For scalars, exclude any protected destination indices so random
        # program generation never overwrites them.
        all_scalar_dests = list(range(0, self.n_scalar))
        protected_set = set(self._protected_scalar_dests)
        writeable_scalar_dests = [i for i in all_scalar_dests if i not in protected_set]

        # Fall back to all registers if protection would leave no writable dest.
        if not writeable_scalar_dests:
            writeable_scalar_dests = all_scalar_dests

        self._dest_ranges = {
            MemoryType.SCALAR: writeable_scalar_dests,
            MemoryType.VECTOR: list(range(0, self.n_vector)),
            MemoryType.MATRIX: list(range(0, self.n_matrix)),
        }
    
    def _get_available_obs_types(self) -> List[MemoryType]:
        """Return list of observation types that have at least one register."""
        available = []
        if self.n_obs_scalar > 0:
            available.append(MemoryType.SCALAR)
        if self.n_obs_vector > 0:
            available.append(MemoryType.VECTOR)
        if self.n_obs_matrix > 0:
            available.append(MemoryType.MATRIX)
        return available
    
    def _choose_obs_register_type(self, src_type: MemoryType, rng: np.random.Generator) -> MemoryType:
        """
        Choose a valid observation register type for extracting the given source type.
        
        Rules:
        - SCALAR can come from: SCALAR, VECTOR (element), MATRIX (element)
        - VECTOR can come from: VECTOR, MATRIX (column)
        - MATRIX can come from: MATRIX
        
        Falls back to available types if preferred type not available.
        """
        if not self._available_obs_types:
            # No observation registers available - return a placeholder
            # (this shouldn't happen if obs_flag is only True when obs exists)
            return MemoryType.SCALAR
        
        # Define compatible obs types for each source type (in preference order)
        compatible = {
            MemoryType.SCALAR: [MemoryType.SCALAR, MemoryType.VECTOR, MemoryType.MATRIX],
            MemoryType.VECTOR: [MemoryType.VECTOR, MemoryType.MATRIX],
            MemoryType.MATRIX: [MemoryType.MATRIX],
        }
        
        # Filter to available types
        valid_types = [t for t in compatible[src_type] if t in self._available_obs_types]
        
        if not valid_types:
            # Fallback: use any available obs type
            return None
        
        return rng.choice(valid_types)
    
    def _get_random_obs_register_index(self, obs_reg_type: MemoryType, rng: np.random.Generator) -> int:
        """Get a random observation register index for the given type."""
        if obs_reg_type == MemoryType.SCALAR:
            return int(rng.integers(0, max(1, self.n_obs_scalar)))
        elif obs_reg_type == MemoryType.VECTOR:
            return int(rng.integers(0, max(1, self.n_obs_vector)))
        else:  # MATRIX
            return int(rng.integers(0, max(1, self.n_obs_matrix)))
    
    def generate_random_instruction(self, rng=None) -> Instruction:
        """
        Generate a random type-safe instruction.
        
        Destination scalar registers listed in ``_protected_scalar_dests`` are
        never chosen as a destination, so the adaptive mutation rate registers
        remain intact during random program generation.
        
        Args:
            rng: numpy random generator (optional)
        
        Returns:
            Valid Instruction with properly configured observation access
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Pick random operation
        op = rng.choice(self.operations)
        
        # Pick source registers based on operation's input types
        source_types = op.input_types()
        source_indices = []
        source_obs_flags = []
        source_obs_register_types = []
        source_obs_register_indices = []
        
        for src_type in source_types:
            # Decide observation flag (50% probability, but only if obs registers exist)
            obs_flag = rng.random() < 0.5 and len(self._available_obs_types) > 0
            source_obs_flags.append(obs_flag)
            
            # Always generate large range index (used as element index for obs, or register index for working)
            source_indices.append(self.get_random_source(src_type, rng))
            
            # Generate observation register type and index
            if obs_flag:
                obs_reg_type = self._choose_obs_register_type(src_type, rng) # gets a valid observation type 
                obs_reg_idx = self._get_random_obs_register_index(obs_reg_type, rng) # gets a valid type 
            else:
                # Placeholder values (not used when obs_flag=False)
                obs_reg_type = MemoryType.SCALAR
                obs_reg_idx = 0
            
            source_obs_register_types.append(obs_reg_type)
            source_obs_register_indices.append(obs_reg_idx)
        
        # Pick destination register (working registers only, respecting protections)
        dest_type = op.output_type()
        dest_index = rng.choice(self._dest_ranges[dest_type])
        
        return Instruction(
            op, dest_type, dest_index, 
            source_types, source_indices, source_obs_flags,
            source_obs_register_types, source_obs_register_indices
        )
    
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

    def get_random_dest(self, dest_type, rng=None):
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