from dataclasses import dataclass
from token import OP
from typing import List
from memory_system import MemoryBank, MemoryType
from operation import Operation

@dataclass
class Instruction:
    operation:Operation
    dest_type: MemoryType
    dest_index: int
    source_types: List[MemoryType]
    source_indices: List[int]

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
    def execute(self, memory:MemoryBank):
        inputs = []
        for src_type, src_idx in zip(self.source_types,self.source_indices):
            if src_idx < 0:
                obs_idx = abs(src_idx) - 1 # - 1 to ensure the 0 value can be obtained - 
                if src_type == MemoryType.SCALAR:
                    inputs.append(float(memory.obs_scalars[obs_idx%memory.n_obs_scalar])) # wrap around babyyyyy
                elif src_type == MemoryType.VECTOR:
                    inputs.append(memory.obs_vectors[obs_idx% memory.n_obs_vector])
                elif src_type == MemoryType.MATRIX:
                    inputs.append(memory.obs_matrices[obs_idx % memory.n_obs_matrix])
            else:
                if src_type == MemoryType.SCALAR:
                    inputs.append(float(memory.scalars[src_idx % memory.n_scalar]))
                elif src_type == MemoryType.VECTOR:
                    inputs.append(memory.vectors[src_idx % memory.n_vector])
                elif src_type == MemoryType.MATRIX:
                    inputs.append(memory.matrices[src_idx % memory.n_matrix])
        
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
        return any(idx < 0 for idx in self.source_indices)
    
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
        for src_type, src_idx in zip(self.source_types, self.source_indices):
            if src_idx < 0:
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
        for t, i in zip(self.source_types, self.source_indices):
            prefix = "o" if i < 0 else ""
            srcs.append(f"{prefix}{type_abbrev[t.value]}{abs(i)}")
        
        return f"{dest}={self.operation.name}({','.join(srcs)})"
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
