"""
Comprehensive tests for generalized observation access in LGP.

Tests the new source_obs_register_types and source_obs_register_indices fields
to ensure they work correctly with various memory configurations.
"""

import numpy as np
import sys
from memory_system import MemoryConfig, MemoryBank, MemoryType
from instruction_set import InstructionSet
from instruction import Instruction
from operation import ScalarAddOp, ScalarMulOp, MINIMAL_SCALAR_OPS
from operators import GeneticOperators
from program import Program


def test_scalar_only_observations():
    """
    TEST 1: Scalar-only observations (Control tasks like CartPole/Pendulum/Acrobot)
    
    Purpose: Verify that when ONLY scalar observations are available (n_obs_vector=0, 
    n_obs_matrix=0), the system correctly:
    - Detects only SCALAR as available obs type
    - Generates instructions that only use SCALAR obs registers
    - Correctly reads scalar observation values during execution
    
    This simulates control tasks where the environment provides a small array of 
    state values (e.g., CartPole's [cart_pos, cart_vel, pole_angle, pole_vel]).
    """
    print("\n" + "="*70)
    print("TEST 1: Scalar-only observations (Control tasks)")
    print("="*70)
    
    # CartPole-like config: 4 scalar observations, NO vector/matrix observations
    cfg = MemoryConfig(
        n_scalar=8, n_vector=0, n_matrix=0,
        n_obs_scalar=4, n_obs_vector=0, n_obs_matrix=0,  # ONLY scalar observations!
        vector_size=4, matrix_shape=(4, 4)
    )
    
    ops = [op() for op in MINIMAL_SCALAR_OPS]
    iset = InstructionSet(ops, cfg)
    
    # Check available obs types - should ONLY be SCALAR
    print(f"Available obs types: {[t.value for t in iset._available_obs_types]}")
    assert iset._available_obs_types == [MemoryType.SCALAR], \
        f"Expected only SCALAR, got {iset._available_obs_types}"
    
    # Generate instructions and verify they ONLY use SCALAR obs type
    rng = np.random.default_rng(42)
    obs_used_count = 0
    for i in range(10):
        instr = iset.generate_random_instruction(rng)
        for j, obs_flag in enumerate(instr.source_obs_flags):
            if obs_flag:
                obs_used_count += 1
                obs_type = instr.source_obs_register_types[j]
                obs_idx = instr.source_obs_register_indices[j]
                print(f"  Instr {i}, src {j}: obs_type={obs_type.value}, obs_idx={obs_idx}")
                # When only SCALAR obs exists, it MUST use SCALAR
                assert obs_type == MemoryType.SCALAR, f"Expected SCALAR obs type, got {obs_type}"
                assert 0 <= obs_idx < cfg.n_obs_scalar, f"obs_idx {obs_idx} out of range [0, {cfg.n_obs_scalar})"
    
    print(f"  Total observation accesses: {obs_used_count}")
    
    # Test execution with scalar observations
    memory = MemoryBank(
        n_scalar=8, n_vector=1, n_matrix=1,
        n_obs_scalar=4, n_obs_vector=0, n_obs_matrix=0,
        vector_size=4, matrix_shape=(4, 4)
    )
    memory.load_observation({
        'scalar': [1.0, 2.0, 3.0, 4.0],  # CartPole-like state
    })
    
    # Create an instruction that reads from scalar observations
    # obs_scalars[0] = 1.0, obs_scalars[2] = 3.0, so result should be 4.0
    instr = Instruction(
        operation=ScalarAddOp(),
        dest_type=MemoryType.SCALAR,
        dest_index=0,
        source_types=[MemoryType.SCALAR, MemoryType.SCALAR],
        source_indices=[0, 0],  # element indices (not used for direct scalar read)
        source_obs_flags=[True, True],
        source_obs_register_types=[MemoryType.SCALAR, MemoryType.SCALAR],
        source_obs_register_indices=[0, 2],  # Read obs_scalars[0] and obs_scalars[2]
    )
    
    print(f"\nExecuting: {instr}")
    instr.execute(memory)
    result = memory.read_scalar(0)
    expected = 1.0 + 3.0  # obs_scalars[0] + obs_scalars[2]
    print(f"Result: {result}, Expected: {expected}")
    assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"
    
    print("✅ Test 1 PASSED: Scalar-only observations work correctly!")


def test_matrix_only_observations():
    """
    TEST 2: Matrix-only observations (Vision tasks like FlappyBird)
    
    Purpose: Verify that when ONLY matrix observations are available (n_obs_scalar=0,
    n_obs_vector=0), the system correctly:
    - Detects only MATRIX as available obs type
    - Generates instructions that use MATRIX obs registers
    - Correctly extracts scalar values from matrix using flat indexing
    
    This simulates vision tasks where the environment provides an image/frame.
    The flat index is computed as: row = flat_idx // width, col = flat_idx % width
    """
    print("\n" + "="*70)
    print("TEST 2: Matrix-only observations (Vision tasks)")
    print("="*70)
    
    # Vision config: only matrix observations
    cfg = MemoryConfig(
        n_scalar=8, n_vector=4, n_matrix=2,
        n_obs_scalar=0, n_obs_vector=0, n_obs_matrix=1,
        vector_size=10, matrix_shape=(5, 5)
    )
    
    ops = [op() for op in MINIMAL_SCALAR_OPS]
    iset = InstructionSet(ops, cfg)
    
    print(f"Available obs types: {[t.value for t in iset._available_obs_types]}")
    assert iset._available_obs_types == [MemoryType.MATRIX], \
        f"Expected only MATRIX, got {iset._available_obs_types}"
    
    # Generate instructions - all obs accesses should use MATRIX
    rng = np.random.default_rng(42)
    for i in range(10):
        instr = iset.generate_random_instruction(rng)
        for j, obs_flag in enumerate(instr.source_obs_flags):
            if obs_flag:
                obs_type = instr.source_obs_register_types[j]
                assert obs_type == MemoryType.MATRIX, f"Expected MATRIX, got {obs_type}"
    
    # Test execution: extract scalar from matrix using flat indexing
    memory = MemoryBank(
        n_scalar=8, n_vector=4, n_matrix=2,
        n_obs_scalar=0, n_obs_vector=0, n_obs_matrix=1,
        vector_size=10, matrix_shape=(5, 5)
    )
    
    # Create a 5x5 matrix: [[0,1,2,3,4], [5,6,7,8,9], [10,11,12,13,14], ...]
    obs_matrix = np.arange(25, dtype=np.float32).reshape(5, 5)
    memory.load_observation({'matrix': [obs_matrix]})
    
    # Instruction to extract elements at flat indices 12 and 0
    # flat_idx=12: row=12//5=2, col=12%5=2 -> obs_matrix[2,2] = 12
    # flat_idx=0:  row=0//5=0, col=0%5=0  -> obs_matrix[0,0] = 0
    instr = Instruction(
        operation=ScalarAddOp(),
        dest_type=MemoryType.SCALAR,
        dest_index=0,
        source_types=[MemoryType.SCALAR, MemoryType.SCALAR],
        source_indices=[12, 0],  # flat indices into matrix
        source_obs_flags=[True, True],
        source_obs_register_types=[MemoryType.MATRIX, MemoryType.MATRIX],
        source_obs_register_indices=[0, 0],
    )
    
    print(f"\nExecuting: {instr}")
    instr.execute(memory)
    result = memory.read_scalar(0)
    expected = 12.0 + 0.0  # obs_matrix[2,2] + obs_matrix[0,0]
    print(f"Result: {result}, Expected: {expected}")
    assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"
    
    print("✅ Test 2 PASSED: Matrix-only observations work correctly!")


def test_vector_only_observations():
    """
    TEST 3: Vector-only observations
    
    Purpose: Verify that when ONLY vector observations are available, the system:
    - Detects only VECTOR as available obs type
    - Correctly extracts scalar elements from vector observations
    
    This simulates sensors that provide 1D arrays (e.g., lidar scans, joint positions).
    The source_indices field specifies which element to extract from the vector.
    """
    print("\n" + "="*70)
    print("TEST 3: Vector-only observations")
    print("="*70)
    
    cfg = MemoryConfig(
        n_scalar=8, n_vector=4, n_matrix=1,
        n_obs_scalar=0, n_obs_vector=2, n_obs_matrix=0,  # ONLY vector observations
        vector_size=10, matrix_shape=(5, 5)
    )
    
    ops = [op() for op in MINIMAL_SCALAR_OPS]
    iset = InstructionSet(ops, cfg)
    
    print(f"Available obs types: {[t.value for t in iset._available_obs_types]}")
    assert iset._available_obs_types == [MemoryType.VECTOR], \
        f"Expected only VECTOR, got {iset._available_obs_types}"
    
    # Test execution: extract scalar from vector
    memory = MemoryBank(
        n_scalar=8, n_vector=4, n_matrix=1,
        n_obs_scalar=0, n_obs_vector=2, n_obs_matrix=1,
        vector_size=10, matrix_shape=(5, 5)
    )
    
    # obs_vectors[0] = [0,1,2,3,4,5,6,7,8,9]
    # obs_vectors[1] = [10,11,12,13,14,15,16,17,18,19]
    obs_vec0 = np.arange(10, dtype=np.float32)
    obs_vec1 = np.arange(10, 20, dtype=np.float32)
    memory.load_observation({'vector': [obs_vec0, obs_vec1]})
    
    # Extract obs_vectors[0][5] = 5 and obs_vectors[1][3] = 13
    instr = Instruction(
        operation=ScalarAddOp(),
        dest_type=MemoryType.SCALAR,
        dest_index=0,
        source_types=[MemoryType.SCALAR, MemoryType.SCALAR],
        source_indices=[5, 3],  # element indices within vectors
        source_obs_flags=[True, True],
        source_obs_register_types=[MemoryType.VECTOR, MemoryType.VECTOR],
        source_obs_register_indices=[0, 1],  # which vector registers
    )
    
    print(f"\nExecuting: {instr}")
    instr.execute(memory)
    result = memory.read_scalar(0)
    expected = 5.0 + 13.0  # obs_vec0[5] + obs_vec1[3]
    print(f"Result: {result}, Expected: {expected}")
    assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"
    
    print("✅ Test 3 PASSED: Vector-only observations work correctly!")


def test_mixed_observation_types():
    """
    TEST 4: Mixed observation types (Multi-sensor robot)
    
    Purpose: Verify that when ALL observation types are available, the system:
    - Correctly detects all three types as available
    - Can mix different observation sources in a single instruction
    - One operand from scalar obs + one operand from matrix obs
    
    This simulates a robot with multiple sensors (e.g., battery level + camera).
    """
    print("\n" + "="*70)
    print("TEST 4: Mixed observation types")
    print("="*70)
    
    cfg = MemoryConfig(
        n_scalar=8, n_vector=4, n_matrix=2,
        n_obs_scalar=3, n_obs_vector=2, n_obs_matrix=1,
        vector_size=10, matrix_shape=(5, 5)
    )
    
    ops = [op() for op in MINIMAL_SCALAR_OPS]
    iset = InstructionSet(ops, cfg)
    
    print(f"Available obs types: {[t.value for t in iset._available_obs_types]}")
    assert len(iset._available_obs_types) == 3, "All 3 obs types should be available"
    
    # Test execution with mixed sources
    memory = MemoryBank(
        n_scalar=8, n_vector=4, n_matrix=2,
        n_obs_scalar=3, n_obs_vector=2, n_obs_matrix=1,
        vector_size=10, matrix_shape=(5, 5)
    )
    
    memory.load_observation({
        'scalar': [100.0, 200.0, 300.0],  # e.g., battery, temp, speed
        'vector': [np.ones(10) * 10, np.ones(10) * 20],  # e.g., lidar scans
        'matrix': [np.ones((5, 5)) * 5]  # e.g., camera image
    })
    
    # Mix sources: one from scalar obs, one from matrix obs
    # obs_scalars[1] = 200.0, obs_matrix[0][0,0] = 5.0
    instr = Instruction(
        operation=ScalarAddOp(),
        dest_type=MemoryType.SCALAR,
        dest_index=0,
        source_types=[MemoryType.SCALAR, MemoryType.SCALAR],
        source_indices=[0, 0],  # element indices
        source_obs_flags=[True, True],
        source_obs_register_types=[MemoryType.SCALAR, MemoryType.MATRIX],
        source_obs_register_indices=[1, 0],  # obs_scalars[1] + obs_matrix[0][0,0]
    )
    
    print(f"\nExecuting: {instr}")
    instr.execute(memory)
    result = memory.read_scalar(0)
    expected = 200.0 + 5.0  # obs_scalars[1] + obs_matrix element
    print(f"Result: {result}, Expected: {expected}")
    assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"
    
    print("✅ Test 4 PASSED: Mixed observation types work correctly!")


def test_obs_flag_false():
    """
    TEST 5: Working register access (obs_flag=False)
    
    Purpose: Verify that when obs_flag=False, the instruction reads from working
    registers (not observation registers), regardless of source_obs_register_types.
    
    This is important because programs mix observation inputs with intermediate
    computations stored in working registers.
    """
    print("\n" + "="*70)
    print("TEST 5: Working register access (obs_flag=False)")
    print("="*70)
    
    cfg = MemoryConfig(
        n_scalar=8, n_vector=4, n_matrix=2,
        n_obs_scalar=4, n_obs_vector=0, n_obs_matrix=1,
        vector_size=10, matrix_shape=(5, 5)
    )
    
    memory = MemoryBank(
        n_scalar=8, n_vector=4, n_matrix=2,
        n_obs_scalar=4, n_obs_vector=1, n_obs_matrix=1,
        vector_size=10, matrix_shape=(5, 5)
    )
    
    # Set working register values (these are evolvable constants)
    memory.write_scalar(0, 10.0)
    memory.write_scalar(1, 20.0)
    
    # Instruction using only working registers (obs_flag=False)
    # The source_obs_register_* fields are IGNORED when obs_flag=False
    instr = Instruction(
        operation=ScalarAddOp(),
        dest_type=MemoryType.SCALAR,
        dest_index=2,
        source_types=[MemoryType.SCALAR, MemoryType.SCALAR],
        source_indices=[0, 1],  # Working register indices
        source_obs_flags=[False, False],  # NOT using observations
        source_obs_register_types=[MemoryType.SCALAR, MemoryType.SCALAR],  # Ignored!
        source_obs_register_indices=[99, 99],  # Ignored! Can be any value
    )
    
    print(f"\nExecuting: {instr}")
    instr.execute(memory)
    result = memory.read_scalar(2)
    expected = 10.0 + 20.0  # scalars[0] + scalars[1]
    print(f"Result: {result}, Expected: {expected}")
    assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"
    
    print("✅ Test 5 PASSED: Working register access works correctly!")


def test_mixed_obs_and_working():
    """
    TEST 6: Mixed observation and working register sources
    
    Purpose: Verify that a single instruction can have one operand from an
    observation register and another operand from a working register.
    
    This is the most common case in LGP - reading environment input (obs)
    and combining it with evolved constants or intermediate values (working).
    Example: output = obs_value * evolved_constant
    """
    print("\n" + "="*70)
    print("TEST 6: Mixed observation and working register sources")
    print("="*70)
    
    cfg = MemoryConfig(
        n_scalar=8, n_vector=0, n_matrix=0,
        n_obs_scalar=4, n_obs_vector=0, n_obs_matrix=1,
        vector_size=4, matrix_shape=(4, 4)
    )
    
    memory = MemoryBank(
        n_scalar=8, n_vector=1, n_matrix=1,
        n_obs_scalar=4, n_obs_vector=1, n_obs_matrix=1,
        vector_size=4, matrix_shape=(4, 4)
    )
    
    memory.load_observation({'scalar': [100.0, 200.0, 300.0, 400.0]})
    memory.write_scalar(0, 5.0)  # Working register = evolved constant
    
    # First source: observation (obs_scalars[2] = 300.0)
    # Second source: working register (scalars[0] = 5.0)
    instr = Instruction(
        operation=ScalarMulOp(),
        dest_type=MemoryType.SCALAR,
        dest_index=1,
        source_types=[MemoryType.SCALAR, MemoryType.SCALAR],
        source_indices=[0, 0],  # elem_idx for obs, reg_idx for working
        source_obs_flags=[True, False],  # First from obs, second from working
        source_obs_register_types=[MemoryType.SCALAR, MemoryType.SCALAR],
        source_obs_register_indices=[2, 0],  # obs_scalars[2], (ignored for working)
    )
    
    print(f"\nExecuting: {instr}")
    instr.execute(memory)
    result = memory.read_scalar(1)
    expected = 300.0 * 5.0  # obs_scalars[2] * scalars[0]
    print(f"Result: {result}, Expected: {expected}")
    assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"
    
    print("✅ Test 6 PASSED: Mixed observation and working sources work correctly!")


def test_mutation_with_obs_sources():
    """
    TEST 7: Genetic operators with observation sources
    
    Purpose: Verify that all genetic operators correctly handle the new 
    source_obs_register_types and source_obs_register_indices fields:
    - micro_mutate: Updates fields correctly for all mutation types
    - macro_mutate: Generates new instruction with valid fields
    - mutate_program: Maintains field consistency across program
    
    This ensures evolution doesn't break due to field mismatches.
    """
    print("\n" + "="*70)
    print("TEST 7: Genetic operators with observation sources")
    print("="*70)
    
    cfg = MemoryConfig(
        n_scalar=8, n_vector=0, n_matrix=0,
        n_obs_scalar=4, n_obs_vector=0, n_obs_matrix=1,
        vector_size=4, matrix_shape=(4, 4)
    )
    
    ops = [op() for op in MINIMAL_SCALAR_OPS]
    iset = InstructionSet(ops, cfg)
    genetic_ops = GeneticOperators(iset, np.random.default_rng(42))
    
    rng = np.random.default_rng(42)
    instr = iset.generate_random_instruction(rng)
    print(f"Initial: {instr}")
    print(f"  obs_reg_types: {[t.value for t in instr.source_obs_register_types]}")
    
    # Test micro mutations (all 4 types: op replace, dest mutate, src mutate, obs mutate)
    for i in range(20):
        instr_copy = iset.generate_random_instruction(rng)
        genetic_ops.micro_mutate(instr_copy, rng)
        # Verify fields still match
        assert len(instr_copy.source_obs_register_types) == len(instr_copy.source_types), \
            f"obs_reg_types mismatch after micro_mutate"
        assert len(instr_copy.source_obs_register_indices) == len(instr_copy.source_types), \
            f"obs_reg_indices mismatch after micro_mutate"
    
    # Test macro mutation (complete replacement)
    new_instr = genetic_ops.macro_mutate(instr, rng)
    assert len(new_instr.source_obs_register_types) == len(new_instr.source_types)
    
    # Test program mutation (mutations + insertions + deletions)
    program = iset.generate_random_program(5, rng)
    for _ in range(10):  # Multiple mutation rounds
        genetic_ops.mutate_program(program, threshold=0.8, rng=rng, max_length=20)
        for inst in program.instructions:
            assert len(inst.source_obs_register_types) == len(inst.source_types), \
                f"Field mismatch in program mutation"
            assert len(inst.source_obs_register_indices) == len(inst.source_types)
    
    print("✅ Test 7 PASSED: Genetic operators work correctly!")


def test_program_copy():
    """
    TEST 8: Program copy with new fields
    
    Purpose: Verify that Program.copy() creates a deep copy that includes
    the new source_obs_register_types and source_obs_register_indices fields.
    
    Deep copy is critical for evolution - offspring must be independent of parents.
    Modifying a copy should not affect the original.
    """
    print("\n" + "="*70)
    print("TEST 8: Program copy with new fields")
    print("="*70)
    
    cfg = MemoryConfig(
        n_scalar=8, n_vector=0, n_matrix=0,
        n_obs_scalar=4, n_obs_vector=0, n_obs_matrix=1,
        vector_size=4, matrix_shape=(4, 4)
    )
    
    ops = [op() for op in MINIMAL_SCALAR_OPS]
    iset = InstructionSet(ops, cfg)
    rng = np.random.default_rng(42)
    
    program = iset.generate_random_program(5, rng)
    program_copy = program.copy()
    
    # Verify values are equal but objects are independent
    for orig, copy in zip(program.instructions, program_copy.instructions):
        # Values should be equal
        assert orig.source_obs_register_types == copy.source_obs_register_types
        assert orig.source_obs_register_indices == copy.source_obs_register_indices
        # But they should be different list objects (deep copy)
        assert orig.source_obs_register_types is not copy.source_obs_register_types
        assert orig.source_obs_register_indices is not copy.source_obs_register_indices
    
    # Modify copy and verify original is unchanged
    if program_copy.instructions and program_copy.instructions[0].source_obs_register_indices:
        original_value = program.instructions[0].source_obs_register_indices[0]
        program_copy.instructions[0].source_obs_register_indices[0] = 999
        assert program.instructions[0].source_obs_register_indices[0] == original_value, \
            "Original was modified when copy was changed!"
    
    print("✅ Test 8 PASSED: Program copy works correctly!")


def test_crossover():
    """
    TEST 9: Crossover with new fields
    
    Purpose: Verify that crossover operations (one-point and two-point) work
    correctly with the new fields. Children should inherit complete instructions
    from parents, including all obs register fields.
    
    Crossover swaps segments of instructions between parents to create offspring.
    """
    print("\n" + "="*70)
    print("TEST 9: Crossover with new fields")
    print("="*70)
    
    cfg = MemoryConfig(
        n_scalar=8, n_vector=0, n_matrix=0,
        n_obs_scalar=4, n_obs_vector=0, n_obs_matrix=1,
        vector_size=4, matrix_shape=(4, 4)
    )
    
    ops = [op() for op in MINIMAL_SCALAR_OPS]
    iset = InstructionSet(ops, cfg)
    genetic_ops = GeneticOperators(iset, np.random.default_rng(42))
    rng = np.random.default_rng(42)
    
    parent1 = iset.generate_random_program(10, rng)
    parent2 = iset.generate_random_program(10, rng)
    
    # Test one-point crossover
    child1, child2 = genetic_ops.one_point_crossover(parent1, parent2, rng)
    
    # Test two-point crossover
    child3, child4 = genetic_ops.two_point_crossover(parent1, parent2, rng)
    
    # Verify all children have valid instructions with consistent fields
    for prog in [child1, child2, child3, child4]:
        for instr in prog.instructions:
            assert len(instr.source_obs_register_types) == len(instr.source_types), \
                "obs_reg_types length mismatch after crossover"
            assert len(instr.source_obs_register_indices) == len(instr.source_types), \
                "obs_reg_indices length mismatch after crossover"
    
    print("✅ Test 9 PASSED: Crossover works correctly!")


def test_edge_case_no_observations():
    """
    TEST 10: Edge case - No observations at all
    
    Purpose: Verify that the system handles the case where there are NO
    observation registers (n_obs_scalar=0, n_obs_vector=0, n_obs_matrix=0).
    
    In this case:
    - _available_obs_types should be empty
    - All generated instructions should have obs_flag=False
    - Programs should only use working registers
    
    This could happen in pure optimization tasks with no environment input.
    """
    print("\n" + "="*70)
    print("TEST 10: Edge case - No observations")
    print("="*70)
    
    cfg = MemoryConfig(
        n_scalar=8, n_vector=4, n_matrix=2,
        n_obs_scalar=0, n_obs_vector=0, n_obs_matrix=0,  # No observations!
        vector_size=10, matrix_shape=(5, 5)
    )
    
    ops = [op() for op in MINIMAL_SCALAR_OPS]
    iset = InstructionSet(ops, cfg)
    
    print(f"Available obs types: {iset._available_obs_types}")
    assert len(iset._available_obs_types) == 0, "Should have no available obs types"
    
    # Generate instructions - ALL should have obs_flag=False
    rng = np.random.default_rng(42)
    for i in range(20):
        instr = iset.generate_random_instruction(rng)
        assert not any(instr.source_obs_flags), \
            f"Instruction {i} has obs_flag=True but no observations exist!"
    
    print("✅ Test 10 PASSED: No observations edge case handled correctly!")


def test_string_representations():
    """
    TEST 11: String representations
    
    Purpose: Verify that all string representation methods work correctly
    with the new fields and produce readable output:
    - __repr__: Human-readable format
    - to_compact_str: Short format for logging
    - to_resolved_str: Shows actual indices after modulo
    
    Good string representations are essential for debugging evolution.
    """
    print("\n" + "="*70)
    print("TEST 11: String representations")
    print("="*70)
    
    cfg = MemoryConfig(
        n_scalar=8, n_vector=0, n_matrix=0,
        n_obs_scalar=4, n_obs_vector=0, n_obs_matrix=1,
        vector_size=4, matrix_shape=(4, 4)
    )
    
    # Create instruction with one obs source and one working source
    instr = Instruction(
        operation=ScalarAddOp(),
        dest_type=MemoryType.SCALAR,
        dest_index=0,
        source_types=[MemoryType.SCALAR, MemoryType.SCALAR],
        source_indices=[100, 200],
        source_obs_flags=[True, False],  # First from obs, second from working
        source_obs_register_types=[MemoryType.SCALAR, MemoryType.SCALAR],
        source_obs_register_indices=[2, 0],
    )
    
    repr_str = repr(instr)
    compact_str = instr.to_compact_str()
    resolved_str = instr.to_resolved_str(cfg)
    
    print(f"repr: {repr_str}")
    print(f"compact: {compact_str}")
    print(f"resolved: {resolved_str}")
    
    # Verify output contains expected information
    assert "obs" in repr_str.lower() or "scalar" in repr_str.lower(), \
        "repr should show observation info"
    assert len(compact_str) > 0, "compact_str should not be empty"
    assert "obs_scalar" in resolved_str, "resolved_str should show obs_scalar"
    
    print("✅ Test 11 PASSED: String representations work correctly!")


def test_full_evolution_simulation():
    """
    TEST 12: Full evolution simulation
    
    Purpose: Run a complete mini-evolution to verify all components work together:
    - Population initialization
    - Program execution with observation loading
    - Crossover between parents
    - Mutation of offspring
    - Multiple generations
    
    This is the ultimate integration test - if evolution runs without crashes,
    all the new observation access code is working correctly.
    """
    print("\n" + "="*70)
    print("TEST 12: Full evolution simulation")
    print("="*70)
    
    # CartPole-like setup (scalar observations only)
    cfg = MemoryConfig(
        n_scalar=8, n_vector=0, n_matrix=0,
        n_obs_scalar=4, n_obs_vector=0, n_obs_matrix=0,  # Scalar-only!
        vector_size=4, matrix_shape=(4, 4)
    )
    
    ops = [op() for op in MINIMAL_SCALAR_OPS]
    iset = InstructionSet(ops, cfg)
    genetic_ops = GeneticOperators(iset, np.random.default_rng(42))
    rng = np.random.default_rng(42)
    
    # Create memory template
    memory = MemoryBank(
        n_scalar=8, n_vector=1, n_matrix=1,
        n_obs_scalar=4, n_obs_vector=0, n_obs_matrix=0,
        vector_size=4, matrix_shape=(4, 4)
    )
    
    # Initialize population
    population = [iset.generate_random_program(5, rng) for _ in range(20)]
    
    # Run 10 generations
    for gen in range(10):
        # Evaluate each individual (execute to check for runtime errors)
        for prog in population:
            mem_copy = memory.copy()
            # Load CartPole-like state
            mem_copy.load_observation({'scalar': [0.1, -0.2, 0.3, -0.4]})
            try:
                prog.execute(mem_copy)
            except Exception as e:
                print(f"ERROR in generation {gen}: {e}")
                print(f"Program: {prog}")
                for i, inst in enumerate(prog.instructions):
                    print(f"  {i}: {inst}")
                raise
        
        # Create next generation via crossover + mutation
        new_pop = []
        for i in range(0, len(population), 2):
            p1, p2 = population[i], population[min(i+1, len(population)-1)]
            c1, c2 = genetic_ops.crossover(p1, p2, rng=rng)
            genetic_ops.mutate_program(c1, threshold=0.3, rng=rng, max_length=20)
            genetic_ops.mutate_program(c2, threshold=0.3, rng=rng, max_length=20)
            new_pop.extend([c1, c2])
        
        population = new_pop[:20]
    
    print(f"Completed 10 generations with {len(population)} individuals")
    print("✅ Test 12 PASSED: Full evolution simulation works correctly!")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("GENERALIZED OBSERVATION ACCESS TEST SUITE")
    print("="*70)
    
    tests = [
        test_scalar_only_observations,
        test_matrix_only_observations,
        test_vector_only_observations,
        test_mixed_observation_types,
        test_obs_flag_false,
        test_mixed_obs_and_working,
        test_mutation_with_obs_sources,
        test_program_copy,
        test_crossover,
        test_edge_case_no_observations,
        test_string_representations,
        test_full_evolution_simulation,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n❌ FAILED: {test.__name__}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*70)
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉")
    else:
        print("\n⚠️ SOME TESTS FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()

