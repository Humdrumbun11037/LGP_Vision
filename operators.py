"""Genetic operators for mutating instructions."""

from __future__ import annotations

from typing import Optional

import numpy as np

from instruction import Instruction
from instruction_set import InstructionSet
from operation import ALL_OPS
from program import Program
from memory_system import MemoryBank, MemoryType


class GeneticOperators:
    """Collection of mutation utilities for LGP instructions."""

    def __init__(
        self,
        instruction_set: InstructionSet,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """Keep references to the instruction set and optional RNG."""
        self.instruction_set = instruction_set
        self._rng = rng

    def micro_mutate(
        self,
        instruction: Instruction,
        rng: Optional[np.random.Generator] = None,
    ) -> Instruction:
        """Perform an in-place micro-mutation on a single instruction.

        Args:
            instruction: Instruction to mutate.
            rng: Optional random number generator.

        Returns:
            The mutated instruction (same instance, mutated in place).
        
        Mutation types:
            0: Replace entire operation
            1: Mutate destination index
            2: Mutate source element index
            3: Mutate observation source (type and/or register index)
        """

        generator = rng or self._rng or np.random.default_rng()

        # Randomly choose which micro mutation to apply.
        mutation = int(generator.integers(0, 4))

        if mutation == 0:
            # Replace the entire operation and refresh dependent fields.
            new_op = self.instruction_set.get_random_operator(generator)
            instruction.operation = new_op
            instruction.dest_type = new_op.output_type()
            instruction.dest_index = self.instruction_set.get_random_dest(
                instruction.dest_type, generator
            )
            instruction.source_types = new_op.input_types()
            
            # Check if any observation types are available
            has_obs = len(self.instruction_set._available_obs_types) > 0
            
            # Generate indices and observation flags
            instruction.source_indices = []
            instruction.source_obs_flags = []
            instruction.source_obs_register_types = []
            instruction.source_obs_register_indices = []
            
            for src_type in instruction.source_types:
                # Observation flag (50% probability, only if obs exists)
                obs_flag = generator.random() < 0.5 and has_obs
                instruction.source_obs_flags.append(obs_flag)
                
                # Element index
                instruction.source_indices.append(
                    self.instruction_set.get_random_source(src_type, generator)
                )
                
                # Observation register type and index
                if obs_flag:
                    obs_reg_type = self.instruction_set._choose_obs_register_type(src_type, generator)
                    obs_reg_idx = self.instruction_set._get_random_obs_register_index(obs_reg_type, generator)
                else:
                    obs_reg_type = MemoryType.SCALAR
                    obs_reg_idx = 0
                
                instruction.source_obs_register_types.append(obs_reg_type)
                instruction.source_obs_register_indices.append(obs_reg_idx)
                
        elif mutation == 1:
            # Mutate only the destination index (keeping type intact).
            instruction.dest_index = self.instruction_set.get_random_dest(
                instruction.dest_type, generator
            )
        elif mutation == 2:
            # Mutate one of the source element indices, if any exist.
            if instruction.source_types:
                src_position = int(generator.integers(0, len(instruction.source_types)))
                instruction.source_indices[src_position] = self.instruction_set.get_random_source(
                    instruction.source_types[src_position], generator
                )
        else:
            # Mutate observation source (type and/or register index)
            # Find positions with obs_flag=True
            obs_positions = [i for i, f in enumerate(instruction.source_obs_flags) if f]
            
            if obs_positions and len(self.instruction_set._available_obs_types) > 0:
                pos = generator.choice(obs_positions)
                src_type = instruction.source_types[pos]
                
                # Randomly choose to mutate type, index, or both
                mutate_what = int(generator.integers(0, 3))
                
                if mutate_what == 0 or mutate_what == 2:
                    # Mutate obs register type
                    new_obs_type = self.instruction_set._choose_obs_register_type(src_type, generator)
                    instruction.source_obs_register_types[pos] = new_obs_type
                    # Also update index to be valid for new type
                    instruction.source_obs_register_indices[pos] = \
                        self.instruction_set._get_random_obs_register_index(new_obs_type, generator)
                
                if mutate_what == 1 or mutate_what == 2:
                    # Mutate obs register index (keeping type)
                    obs_reg_type = instruction.source_obs_register_types[pos]
                    instruction.source_obs_register_indices[pos] = \
                        self.instruction_set._get_random_obs_register_index(obs_reg_type, generator)

        return instruction

    def macro_mutate(self,instruction: Instruction, rng = None):
        """Return a brand new random instruction (macro mutation)."""
        if rng is None:
            rng = np.random.default_rng()
        instruction = self.instruction_set.generate_random_instruction(rng)
        return instruction

    def add_instruction_mutate(self, program, index, rng = None):
        """Insert a new instruction right after the supplied index."""
        if rng is None:
            rng = np.random.default_rng()
        new_instruction = self.instruction_set.generate_random_instruction(rng)
        insert_at = min(index + 1, len(program.instructions))
        program.instructions.insert(insert_at, new_instruction)
        return program

    def delete_instruction_mutate(self, program, index):
        """Remove the instruction at index (if program has > 1 instruction)."""
        if len(program.instructions) <= 1:
            return program
        remove_at = min(index, len(program.instructions) - 1)
        program.instructions.pop(remove_at)
        return program

    def mutate_program(self, program, threshold, rng = None, max_length = None):
        """Walk every instruction and mutate when random() <= threshold."""
        if rng is None:
            rng = np.random.default_rng()
        i = 0
        while i < len(program.instructions):
            if rng.random() <= threshold:
                mutation_type = int(rng.integers(0,4))
                if mutation_type == 0:
                    self.micro_mutate(program.instructions[i], rng)
                elif mutation_type == 1:
                    program.instructions[i] = self.macro_mutate(program.instructions[i], rng)
                elif mutation_type == 2:
                    if max_length is None or len(program.instructions) < max_length:
                        self.add_instruction_mutate(program, i, rng)
                else:
                    self.delete_instruction_mutate(program, i)
                    i -= 1
            i += 1
        if len(program.instructions) == 0:
            # Guarantee program never becomes empty.
            program.instructions.append(self.instruction_set.generate_random_instruction(rng))
        return program

    def one_point_crossover(self, parent1: Program, parent2: Program, rng= None):
        if rng is None:
            rng = np.random.default_rng()
        p1 = parent1.copy()
        p2 = parent2.copy()
        min_len = min(len(p1), len(p2))
        if min_len < 2:
            return p1, p2
        cut = int(rng.integers(1, min_len))
        c1_instrs = p1.instructions[:cut] + p2.instructions[cut:]
        c2_instrs = p2.instructions[:cut] + p1.instructions[cut:]
        c1 = Program(c1_instrs)
        c2 = Program(c2_instrs)
        return c1, c2


    def two_point_crossover(self, parent1: Program, parent2: Program, rng= None):
        if rng is None:
            rng = np.random.default_rng()
        p1 = parent1.copy()
        p2 = parent2.copy()
        min_len = min(len(p1), len(p2))
        if min_len < 2:
            return p1, p2
        # choose two cut points on each parent
        c11 = int(rng.integers(0, len(p1)))
        c12 = int(rng.integers(c11, len(p1)))
        c21 = int(rng.integers(0, len(p2)))
        c22 = int(rng.integers(c21, len(p2)))
        c1_instrs = p1.instructions[:c11] + p2.instructions[c21:c22] + p1.instructions[c12:]
        c2_instrs = p2.instructions[:c21] + p1.instructions[c11:c12] + p2.instructions[c22:]
        c1 = Program(c1_instrs)
        c2 = Program(c2_instrs)
        return c1, c2 
    # roughly 161 Times speed up over naive 
    def mutate_constants(self, memory: MemoryBank, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        if len(memory.scalars):
            factors = rng.uniform(0.5, 2.0, size=memory.scalars.shape)
            signs = np.where(rng.random(memory.scalars.shape) < 0.1, -1.0, 1.0)
            memory.scalars *= factors * signs
        if len(memory.vectors):
            factors = rng.uniform(0.5, 2.0, size=memory.vectors.shape)
            signs = np.where(rng.random(memory.vectors.shape) < 0.1, -1.0, 1.0)
            memory.vectors *= factors * signs
        if len(memory.matrices):
            factors = rng.uniform(0.5, 2.0, size=memory.matrices.shape)
            signs = np.where(rng.random(memory.matrices.shape) < 0.1, -1.0, 1.0)
            memory.matrices *= factors * signs

    def mutate_constants_naive(self, memory: MemoryBank, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        dis1 = rng.uniform
        dis2 = rng.uniform
        for idx in range(len(memory.scalars)):
            factor = dis1(0.5, 2.0)
            memory.scalars[idx] *= factor
            if dis2(0.0, 1.0) <= 0.1:
                memory.scalars[idx] *= -1
        for vec in memory.vectors:
            for j in range(len(vec)):
                factor = dis1(0.5, 2.0)
                vec[j] *= factor
                if dis2(0.0, 1.0) <= 0.1:
                    vec[j] *= -1
        for mat in memory.matrices:
            rows, cols = mat.shape
            for r in range(rows):
                for c in range(cols):
                    factor = dis1(0.5, 2.0)
                    mat[r, c] *= factor
                    if dis2(0.0, 1.0) <= 0.1:
                        mat[r, c] *= -1

    def crossover(self, parent1: Program, parent2: Program, threshold =0.9,rng = None):
        if rng is None:
            rng = np.random.default_rng()
        slc = int(rng.integers(0,2))
        if rng.random() > threshold:
            return parent1.copy(), parent2.copy()
        if slc == 0:
            c1,c2 = self.one_point_crossover(parent1,parent2,rng)
        else:
            c1, c2 = self.two_point_crossover(parent1, parent2,rng)
        return c1, c2




if __name__ == "__main__":
    from operation import ScalarAddOp, ScalarMulOp
    from memory_system import MemoryBank, MemoryType, MemoryConfig

    memory_config = MemoryConfig(
        n_scalar=8,
        n_vector=8,
        n_matrix=8,
        n_obs_scalar=0,
        n_obs_vector=0,
        n_obs_matrix=1,  # Need at least 1 observation matrix for matrix observation access
        vector_size=5,
        matrix_shape=(9, 9),
    )

    operations = [op() for op in ALL_OPS]
    instr_set = InstructionSet(operations, memory_config)
    ops = GeneticOperators(instr_set, np.random.default_rng(0))

    instruction = Instruction(
        operation=ScalarAddOp(),
        dest_type=MemoryType.SCALAR,
        dest_index=0,
        source_types=[MemoryType.SCALAR, MemoryType.SCALAR],
        source_indices=[0, 1],
        source_obs_flags=[False, False],
        source_obs_register_types=[MemoryType.SCALAR, MemoryType.SCALAR],
        source_obs_register_indices=[0, 0],
    )

    print("Initial instruction:", instruction.to_resolved_str(memory_config))

    print("\n--- Micro Mutation Samples ---")
    for _ in range(3):
        before_repr = instruction.to_resolved_str(memory_config)
        ops.micro_mutate(instruction)
        print("Before:", before_repr)
        print("After: ", instruction.to_resolved_str(memory_config))
        print("Valid?:", instruction.is_valid())
        print()

    print("--- Macro Mutation Samples ---")
    for _ in range(3):
        print("old Instruction:", instruction.to_resolved_str(memory_config))
        new_instr = ops.macro_mutate(instruction)
        print("New instruction:", new_instr.to_resolved_str(memory_config))
        print("Valid?:", new_instr.is_valid())
        print()

    program = Program([
        Instruction(
            operation=ScalarAddOp(),
            dest_type=MemoryType.SCALAR,
            dest_index=0,
            source_types=[MemoryType.SCALAR, MemoryType.SCALAR],
            source_indices=[0, 1],
            source_obs_flags=[False, False],
            source_obs_register_types=[MemoryType.SCALAR, MemoryType.SCALAR],
            source_obs_register_indices=[0, 0],
        )
    ])

    print("--- Program Mutation Samples ---")
    for _ in range(10):
        print("Program before:")
        for idx, instr in enumerate(program.instructions):
            print(f"  {idx}: {instr.to_resolved_str(memory_config)}")
        ops.mutate_program(program, threshold=1.0)
        print("Program after:")
        for idx, instr in enumerate(program.instructions):
            print(f"  {idx}: {instr.to_resolved_str(memory_config)}")
        print()

    # Benchmark constant mutation implementations
    import time
    rng = np.random.default_rng(123)
    big_memory = MemoryBank(
        n_scalar=2048,
        n_vector=256,
        n_matrix=64,
        n_obs_scalar=0,
        n_obs_vector=0,
        n_obs_matrix=0,
        vector_size=64,
        matrix_shape=(32, 32),
    )

    vec_memory = big_memory.copy()
    naive_memory = big_memory.copy()

    start = time.perf_counter()
    ops.mutate_constants(vec_memory, rng)
    vec_time = time.perf_counter() - start

    start = time.perf_counter()
    ops.mutate_constants_naive(naive_memory, rng)
    naive_time = time.perf_counter() - start

    print("Vectorized constant mutation time: %.6f s" % vec_time)
    print("Naive constant mutation time:      %.6f s" % naive_time)
    if vec_time > 0:
        print("Speedup (naive / vectorized):      %.2fx" % (naive_time / vec_time)) 