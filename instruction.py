from dataclasses import dataclass
from token import OP
from typing import List
import numpy as np
from memory_system import MemoryBank, MemoryType, MemoryConfig
from operation import Operation

@dataclass
class Instruction:
    operation: Operation
    dest_type: MemoryType
    dest_index: int
    source_types: List[MemoryType]
    source_indices: List[int]
    source_obs_flags: List[bool]
    # NEW: Explicit observation source specification
    source_obs_register_types: List[MemoryType]   # Which obs register type (SCALAR/VECTOR/MATRIX)
    source_obs_register_indices: List[int]        # Which register of that type

    def __post_init__(self):
        """Validate instruction after creation"""
        if self.dest_index < 0:
            raise ValueError(
                f"Destination index must be non-negative (got {self.dest_index}). "
                "Cannot write to observation registers."
            )
        
        n_sources = len(self.source_types)
        
        if len(self.source_indices) != n_sources:
            raise ValueError(
                f"Mismatch: {n_sources} source types but "
                f"{len(self.source_indices)} source indices"
            )
        
        if len(self.source_obs_flags) != n_sources:
            raise ValueError(
                f"Mismatch: {n_sources} source types but "
                f"{len(self.source_obs_flags)} source observation flags"
            )
        
        if len(self.source_obs_register_types) != n_sources:
            raise ValueError(
                f"Mismatch: {n_sources} source types but "
                f"{len(self.source_obs_register_types)} source obs register types"
            )
        
        if len(self.source_obs_register_indices) != n_sources:
            raise ValueError(
                f"Mismatch: {n_sources} source types but "
                f"{len(self.source_obs_register_indices)} source obs register indices"
            )
    def execute(self, memory: MemoryBank):
        """
        Execute instruction with generalized observation access.
        
        When obs_flag=True, uses source_obs_register_types and source_obs_register_indices
        to determine which observation register to read from, with support for:
        - SCALAR obs → SCALAR output (direct read)
        - VECTOR obs → SCALAR output (element extraction using source_indices)
        - MATRIX obs → SCALAR output (flat index extraction using source_indices)
        - VECTOR obs → VECTOR output (direct read)
        - MATRIX obs → VECTOR output (column extraction using source_indices)
        - MATRIX obs → MATRIX output (direct read)
        """
        inputs = []
        
        for i, (src_type, src_idx, obs_flag) in enumerate(
            zip(self.source_types, self.source_indices, self.source_obs_flags)
        ):
            if obs_flag:
                # Observation access - use explicit obs register type and index
                obs_reg_type = self.source_obs_register_types[i] # gets the type of obs its accessing - a valid obs type it can access 
                obs_reg_idx = self.source_obs_register_indices[i] # gets index of that its accessing 
                elem_idx = src_idx  # Reuse source_indices as element index for extraction
                
                if obs_reg_type == MemoryType.SCALAR:
                    # Direct scalar observation read
                    if memory.n_obs_scalar > 0:
                        value = memory.obs_scalars[obs_reg_idx % memory.n_obs_scalar]
                        inputs.append(float(value))
                    else:
                        # Fallback: return 0.0 if no scalar observations available
                        raise ValueError(
                                f"Cannot read from scalar observation register {obs_reg_idx}: "
                                f"no scalar observations available (n_obs_scalar={memory.n_obs_scalar})")
                    
                elif obs_reg_type == MemoryType.VECTOR:
                    if memory.n_obs_vector > 0:
                        obs_vec = memory.obs_vectors[obs_reg_idx % memory.n_obs_vector]
                        if src_type == MemoryType.SCALAR:
                            # Extract element from vector
                            inputs.append(float(obs_vec[elem_idx % memory.vector_size])) # gets an index of the element 
                        else:  # VECTOR
                            # Read whole vector
                            inputs.append(obs_vec.copy())
                    else:
                        raise ValueError(
                                    f"Cannot read from vector observation register {obs_reg_idx}: "
                                    f"no vector observations available (n_obs_vector={memory.n_obs_vector}). "
                                    f"Attempted to extract {src_type.value} from vector observation."
                                )
                        
                elif obs_reg_type == MemoryType.MATRIX:
                    if memory.n_obs_matrix > 0:
                        obs_mat = memory.obs_matrices[obs_reg_idx % memory.n_obs_matrix]
                        h, w = memory.matrix_shape
                        
                        if src_type == MemoryType.SCALAR:
                            # Extract element from matrix (flat index → row, col)
                            flat_idx = elem_idx % (h * w) if h * w > 0 else 0
                            row, col = flat_idx // w if w > 0 else 0, flat_idx % w if w > 0 else 0
                            inputs.append(float(obs_mat[row, col]))
                        elif src_type == MemoryType.VECTOR:
                            # Extract column from matrix
                            col_idx = elem_idx % w if w > 0 else 0
                            inputs.append(obs_mat[:, col_idx].copy())
                        else:  # MATRIX
                            # Read whole matrix
                            inputs.append(obs_mat.copy())
                    else:
                        # Fallback
                         raise ValueError(
                            f"Cannot read from matrix observation register {obs_reg_idx}: "
                            f"no matrix observations available (n_obs_matrix={memory.n_obs_matrix}). "
                            f"Attempted to extract {src_type.value} from matrix observation."
                        )
            else:
                # Working register access (unchanged)
                if src_type == MemoryType.SCALAR:
                    inputs.append(float(memory.scalars[src_idx % memory.n_scalar]))
                elif src_type == MemoryType.VECTOR:
                    inputs.append(memory.vectors[src_idx % memory.n_vector].copy())
                elif src_type == MemoryType.MATRIX:
                    inputs.append(memory.matrices[src_idx % memory.n_matrix].copy())
        
        result = self.operation.execute(*inputs)

        if self.dest_type == MemoryType.SCALAR:
            memory.write_scalar(self.dest_index, result)
        elif self.dest_type == MemoryType.VECTOR:
            memory.write_vector(self.dest_index, result)
        elif self.dest_type == MemoryType.MATRIX:
            memory.write_matrix(self.dest_index, result)

    def is_valid(self) -> bool:
        """
        Check if instruction is type-safe.
        
        Validates:
        1. Operation's output type matches destination type
        2. Operation's input types match source types
        3. Destination is non-negative (working register)
        """
        return (
            self.dest_index >= 0 and
            self.operation.output_type() == self.dest_type and
            self.operation.input_types() == self.source_types
        )
    
    def uses_observation(self) -> bool:
        """Check if this instruction reads from any observation register"""
        return any(self.source_obs_flags)
    
    def get_read_registers(self) -> List[tuple]:
        """
        Get all registers this instruction reads from.
        
        Returns:
            List of (MemoryType, index) tuples
        """
        return list(zip(self.source_types, self.source_indices))
    
    def get_write_register(self) -> tuple:
        """
        Get the register this instruction writes to.
        
        Returns:
            (MemoryType, index) tuple
        """
        return (self.dest_type, self.dest_index)
    
    def __repr__(self) -> str:
        """Human-readable representation"""
        # Format source operands
        src_parts = []
        for i, (src_type, src_idx, obs_flag) in enumerate(
            zip(self.source_types, self.source_indices, self.source_obs_flags)
        ):
            if obs_flag:
                # Observation register - show obs type and register
                obs_reg_type = self.source_obs_register_types[i]
                obs_reg_idx = self.source_obs_register_indices[i]
                src_parts.append(f"obs_{obs_reg_type.value}[{obs_reg_idx}]→{src_type.value}[{src_idx}]")
            else:
                # Working register
                src_parts.append(f"{src_type.value}[{src_idx}]")
        
        src_str = ", ".join(src_parts)
        
        return f"{self.dest_type.value}[{self.dest_index}] = {self.operation.name}({src_str})"
    
    def to_compact_str(self) -> str:
        """Compact string representation for logging"""
        type_abbrev = {'scalar': 's', 'vector': 'v', 'matrix': 'm'}
        
        dest = f"{type_abbrev[self.dest_type.value]}{self.dest_index}"
        
        srcs = []
        for idx, (t, i, obs_flag) in enumerate(
            zip(self.source_types, self.source_indices, self.source_obs_flags)
        ):
            if obs_flag:
                obs_reg_type = self.source_obs_register_types[idx]
                obs_reg_idx = self.source_obs_register_indices[idx]
                # Format: o<obs_type><obs_reg>:<output_type><elem_idx>
                srcs.append(f"o{type_abbrev[obs_reg_type.value]}{obs_reg_idx}:{type_abbrev[t.value]}{i}")
            else:
                srcs.append(f"{type_abbrev[t.value]}{i}")
        
        return f"{dest}={self.operation.name}({','.join(srcs)})"
    
    def to_resolved_str(self, memory_config: MemoryConfig) -> str:
        """
        String representation showing resolved indices (after modulo operations).
        
        Useful for debugging to see which actual registers are being accessed.
        
        Args:
            memory_config: Memory configuration to compute modulo operations
        
        Returns:
            String showing both raw indices and resolved indices
        """
        h, w = memory_config.matrix_shape
        
        # Helper function to safely compute modulo (handles zero case)
        def safe_mod(value: int, divisor: int) -> str:
            if divisor == 0:
                return f"{value}?0"  # Show raw value with ?0 to indicate no registers
            return str(value % divisor)
        
        # Format source operands with resolved indices
        src_parts = []
        for i, (src_type, src_idx, obs_flag) in enumerate(
            zip(self.source_types, self.source_indices, self.source_obs_flags)
        ):
            if obs_flag:
                # Observation access - use explicit obs register type and index
                obs_reg_type = self.source_obs_register_types[i]
                obs_reg_idx = self.source_obs_register_indices[i]
                elem_idx = src_idx
                
                if obs_reg_type == MemoryType.SCALAR:
                    resolved_reg = safe_mod(obs_reg_idx, memory_config.n_obs_scalar)
                    src_parts.append(f"obs_scalar[{obs_reg_idx}→{resolved_reg}]")
                    
                elif obs_reg_type == MemoryType.VECTOR:
                    resolved_reg = safe_mod(obs_reg_idx, memory_config.n_obs_vector)
                    if src_type == MemoryType.SCALAR:
                        resolved_elem = safe_mod(elem_idx, memory_config.vector_size)
                        src_parts.append(f"obs_vector[{obs_reg_idx}→{resolved_reg}][{elem_idx}→{resolved_elem}]")
                    else:
                        src_parts.append(f"obs_vector[{obs_reg_idx}→{resolved_reg}]")
                        
                elif obs_reg_type == MemoryType.MATRIX:
                    resolved_reg = safe_mod(obs_reg_idx, memory_config.n_obs_matrix)
                    if src_type == MemoryType.SCALAR:
                        # Wrap flat index and compute row/col
                        flat_idx = elem_idx % (h * w) if h * w > 0 else 0
                        row = flat_idx // w if w > 0 else 0
                        col = flat_idx % w if w > 0 else 0
                        # Show both original flat index and wrapped row/col for clarity
                        src_parts.append(f"obs_matrix[{obs_reg_idx}→{resolved_reg}][{elem_idx}→({row},{col})]")
                    elif src_type == MemoryType.VECTOR:
                        resolved_col = safe_mod(elem_idx, w)
                        src_parts.append(f"obs_matrix[{obs_reg_idx}→{resolved_reg}][:,{elem_idx}→{resolved_col}]")
                    else:
                        src_parts.append(f"obs_matrix[{obs_reg_idx}→{resolved_reg}]")
            else:
                # Register access - compute resolved index based on register count
                if src_type == MemoryType.SCALAR:
                    resolved = safe_mod(src_idx, memory_config.n_scalar)
                    src_parts.append(f"{src_type.value}[{src_idx}→{resolved}]")
                elif src_type == MemoryType.VECTOR:
                    resolved = safe_mod(src_idx, memory_config.n_vector)
                    src_parts.append(f"{src_type.value}[{src_idx}→{resolved}]")
                elif src_type == MemoryType.MATRIX:
                    resolved = safe_mod(src_idx, memory_config.n_matrix)
                    src_parts.append(f"{src_type.value}[{src_idx}→{resolved}]")
        
        # Resolve destination index
        if self.dest_type == MemoryType.SCALAR:
            dest_resolved = safe_mod(self.dest_index, memory_config.n_scalar)
        elif self.dest_type == MemoryType.VECTOR:
            dest_resolved = safe_mod(self.dest_index, memory_config.n_vector)
        else:  # MATRIX
            dest_resolved = safe_mod(self.dest_index, memory_config.n_matrix)
        
        src_str = ", ".join(src_parts)
        return f"{self.dest_type.value}[{self.dest_index}→{dest_resolved}] = {self.operation.name}({src_str})"
# if __name__ == "__main__":
#     from Operations import ScalarAddOp, VectorDotProductOp, MatrixMeanOp
#     import numpy as np
    
#     print("="*60)
#     print("INSTRUCTION TESTS")
#     print("="*60)
    
#     # Create memory
#     memory = MemoryBank(
#         n_scalar=10,
#         n_vector=5,
#         n_matrix=2,
#         n_obs_scalar=2,
#         n_obs_vector=1,
#         n_obs_matrix=1,
#         vector_size=10,
#         matrix_shape=(20, 20)
#     )
    
#     # Load observations
#     memory.load_observation({
#         'scalar': [5.0, 10.0],
#         'vector': [np.arange(10)],
#         'matrix': [np.ones((20, 20)) * 3]
#     })
    
#     # Set up some constants in working registers
#     memory.write_scalar(0, 0.0)   # Constant: 0
#     memory.write_scalar(1, 1.0)   # Constant: 1
#     memory.write_scalar(2, 2.0)   # Constant: 2
    
#     print("\nMemory state:")
#     print(f"  obs_scalar[-1] = {memory.read_scalar(-1)}")
#     print(f"  obs_scalar[-2] = {memory.read_scalar(-2)}")
#     print(f"  scalar[0] (const) = {memory.read_scalar(0)}")
#     print(f"  scalar[1] (const) = {memory.read_scalar(1)}")
#     print(f"  scalar[2] (const) = {memory.read_scalar(2)}")
    
#     # Test 1: Scalar instruction with observation
#     print("\n--- Test 1: scalar[5] = ADD(obs_scalar[-1], obs_scalar[-2]) ---")
#     instr1 = Instruction(
#         operation=ScalarAddOp(),
#         dest_type=MemoryType.SCALAR,
#         dest_index=5,
#         source_types=[MemoryType.SCALAR, MemoryType.SCALAR],
#         source_indices=[-1, -2]
#     )
#     print(f"Instruction: {instr1}")
#     print(f"Valid: {instr1.is_valid()}")
#     print(f"Uses observation: {instr1.uses_observation()}")
    
#     instr1.execute(memory)
#     print(f"Result: scalar[5] = {memory.read_scalar(5)}")
    
#     # Test 2: Mix observation and working registers
#     print("\n--- Test 2: scalar[6] = ADD(scalar[5], scalar[2]) ---")
#     instr2 = Instruction(
#         operation=ScalarAddOp(),
#         dest_type=MemoryType.SCALAR,
#         dest_index=6,
#         source_types=[MemoryType.SCALAR, MemoryType.SCALAR],
#         source_indices=[5, 2]  # Working registers
#     )
#     print(f"Instruction: {instr2}")
#     instr2.execute(memory)
#     print(f"Result: scalar[6] = {memory.read_scalar(6)}")
    
#     # Test 3: Vector instruction
#     print("\n--- Test 3: scalar[7] = VECTOR_DOT(obs_vector[-1], obs_vector[-1]) ---")
#     instr3 = Instruction(
#         operation=VectorDotProductOp(),
#         dest_type=MemoryType.SCALAR,
#         dest_index=7,
#         source_types=[MemoryType.VECTOR, MemoryType.VECTOR],
#         source_indices=[-1, -1]  # Dot product with itself
#     )
#     print(f"Instruction: {instr3}")
#     instr3.execute(memory)
#     print(f"Result: scalar[7] = {memory.read_scalar(7)}")
    
#     # Test 4: Matrix to scalar
#     print("\n--- Test 4: scalar[8] = MATRIX_MEAN(obs_matrix[-1]) ---")
#     instr4 = Instruction(
#         operation=MatrixMeanOp(),
#         dest_type=MemoryType.SCALAR,
#         dest_index=8,
#         source_types=[MemoryType.MATRIX],
#         source_indices=[-1]
#     )
#     print(f"Instruction: {instr4}")
#     instr4.execute(memory)
#     print(f"Result: scalar[8] = {memory.read_scalar(8)}")
    
#     # Test 5: Compact string representation
#     print("\n--- Test 5: Compact Representation ---")
#     print(f"instr1: {instr1.to_compact_str()}")
#     print(f"instr2: {instr2.to_compact_str()}")
#     print(f"instr3: {instr3.to_compact_str()}")
#     print(f"instr4: {instr4.to_compact_str()}")
    
#     # Test 6: Invalid instruction (writing to observation)
#     print("\n--- Test 6: Invalid Instruction (write to obs) ---")
#     try:
#         invalid_instr = Instruction(
#             operation=ScalarAddOp(),
#             dest_type=MemoryType.SCALAR,
#             dest_index=-1,  # Trying to write to observation!
#             source_types=[MemoryType.SCALAR, MemoryType.SCALAR],
#             source_indices=[0, 1]
#         )
#         print("ERROR: Should have raised ValueError!")
#     except ValueError as e:
#         print(f"✅ Validation working: {e}")
    
#     # Test 7: Get read/write registers
#     print("\n--- Test 7: Dependency Analysis ---")
#     print(f"instr1 reads from: {instr1.get_read_registers()}")
#     print(f"instr1 writes to: {instr1.get_write_register()}")
#     print(f"instr2 reads from: {instr2.get_read_registers()}")
#     print(f"instr2 writes to: {instr2.get_write_register()}")
    
#     print("\n✅ All tests passed!")
