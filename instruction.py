from dataclasses import dataclass
from token import OP
from typing import List
from memory_system import MemoryBank, MemoryType, MemoryConfig
from operation import Operation

@dataclass
class Instruction:
    operation:Operation
    dest_type: MemoryType
    dest_index: int
    source_types: List[MemoryType]
    source_indices: List[int]
    source_obs_flags: List[bool]

    def __post_init__(self):
        """Validate instruction after creation"""
        if self.dest_index < 0:
            raise ValueError(
                f"Destination index must be non-negative (got {self.dest_index}). "
                "Cannot write to observation registers."
            )
        
        if len(self.source_types) != len(self.source_indices):
            raise ValueError(
                f"Mismatch: {len(self.source_types)} source types but "
                f"{len(self.source_indices)} source indices"
            )
        
        if len(self.source_types) != len(self.source_obs_flags):
            raise ValueError(
                f"Mismatch: {len(self.source_types)} source types but "
                f"{len(self.source_obs_flags)} source observation flags"
            )
    def execute(self, memory:MemoryBank):
        inputs = []
        h, w = memory.matrix_shape  # Matrix dimensions (e.g., 37, 37)
        obs_matrix = memory.obs_matrices[0]  # Single observation matrix
        
        for src_type, src_idx, obs_flag in zip(self.source_types, self.source_indices, self.source_obs_flags):
            if obs_flag:
                # Observation matrix access: interpret index based on matrix dimensions
                if src_type == MemoryType.SCALAR:
                    flat_idx = src_idx % (h * w)  # % 1369 for 37x37
                    row = flat_idx // w
                    col = flat_idx % w
                    inputs.append(float(obs_matrix[row, col]))
                elif src_type == MemoryType.VECTOR:
                    col_idx = src_idx % w  # % 37 for 37x37
                    inputs.append(obs_matrix[:, col_idx].copy())  # Column vector
                elif src_type == MemoryType.MATRIX:
                    obs_idx = src_idx % memory.n_obs_matrix  # % 1 = 0 currently
                    inputs.append(memory.obs_matrices[obs_idx].copy())
            else:
                # Register access: interpret index based on register counts
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
        for src_type, src_idx, obs_flag in zip(self.source_types, self.source_indices, self.source_obs_flags):
            if obs_flag:
                # Observation register
                src_parts.append(f"obs_{src_type.value}[{src_idx}]")
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
        for t, i, obs_flag in zip(self.source_types, self.source_indices, self.source_obs_flags):
            prefix = "o" if obs_flag else ""
            srcs.append(f"{prefix}{type_abbrev[t.value]}{i}")
        
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
        for src_type, src_idx, obs_flag in zip(self.source_types, self.source_indices, self.source_obs_flags):
            if obs_flag:
                # Observation access - compute resolved index based on type
                if src_type == MemoryType.SCALAR:
                    resolved = safe_mod(src_idx, h * w)
                    src_parts.append(f"obs_{src_type.value}[{src_idx}→{resolved}]")
                elif src_type == MemoryType.VECTOR:
                    resolved = safe_mod(src_idx, w) if w > 0 else f"{src_idx}?0"
                    src_parts.append(f"obs_{src_type.value}[{src_idx}→col{resolved}]")
                elif src_type == MemoryType.MATRIX:
                    # For matrix observation, we always return the entire observation matrix
                    # The index is modulo by n_obs_matrix (typically 1, so always 0 = full matrix)
                    if memory_config.n_obs_matrix > 0:
                        obs_idx = src_idx % memory_config.n_obs_matrix
                        src_parts.append(f"obs_{src_type.value}[{src_idx}→full_matrix{obs_idx}]")
                    else:
                        src_parts.append(f"obs_{src_type.value}[{src_idx}→?0]")
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
