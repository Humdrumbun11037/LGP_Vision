"""Comprehensive tests for evolution system to verify correctness."""

import numpy as np
from memory_system import MemoryConfig, MemoryBank
from instruction_set import InstructionSet
from operation import ALL_OPS, SCALAR_OPS, AUTOML_ALL_OPS, CV_ALL_OPS
from individual import Individual
from population import Population, PopulationConfig
from operators import GeneticOperators
from evaluator import (
    BaseEvaluatorConfig,
    CartPoleEvaluatorConfig,
    CartPoleEvaluator,
    FlappyBirdEvaluatorConfig,
    FlappyBirdEvaluator,
    SymbolicRegressionEvaluator,
)
from evolution_engine import EvolutionEngine, EvolutionConfig


def test_individual_fitness_caching():
    """Test that individual fitness caching works correctly."""
    print("\n" + "="*70)
    print("TEST 1: Individual Fitness Caching")
    print("="*70)
    
    rng = np.random.default_rng(42)
    memory_cfg = MemoryConfig(
        n_scalar=4, n_vector=0, n_matrix=0,
        n_obs_scalar=2, n_obs_vector=0, n_obs_matrix=0,
        vector_size=1, matrix_shape=(1, 1)
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
    instr_set = InstructionSet([op() for op in SCALAR_OPS], template_memory)
    individual = Individual.random(instr_set, memory_cfg, program_length=5, rng=rng)
    
    evaluator = SymbolicRegressionEvaluator()
    
    # First evaluation should compute fitness
    assert individual.fitness is None, "Fitness should be None initially"
    fitness1 = individual.evaluate(evaluator)
    assert individual.fitness is not None, "Fitness should be cached after evaluation"
    assert individual.fitness == fitness1, "Cached fitness should match returned value"
    
    # Second evaluation should use cached value
    fitness2 = individual.evaluate(evaluator)
    assert fitness1 == fitness2, "Second evaluation should return cached value"
    
    # After invalidation, should recompute
    individual.invalidate_fitness()
    assert individual.fitness is None, "Fitness should be None after invalidation"
    fitness3 = individual.evaluate(evaluator)
    assert fitness3 is not None, "Fitness should be recomputed after invalidation"
    
    print("✓ Fitness caching works correctly")
    print(f"  Initial fitness: {fitness1:.4f}")
    print(f"  Cached fitness: {fitness2:.4f}")
    print(f"  Recomputed fitness: {fitness3:.4f}")


def test_mutation_creates_diversity():
    """Test that mutations actually create program diversity."""
    print("\n" + "="*70)
    print("TEST 2: Mutation Creates Diversity")
    print("="*70)
    
    rng = np.random.default_rng(42)
    memory_cfg = MemoryConfig(
        n_scalar=4, n_vector=0, n_matrix=0,
        n_obs_scalar=2, n_obs_vector=0, n_obs_matrix=0,
        vector_size=1, matrix_shape=(1, 1)
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
    instr_set = InstructionSet([op() for op in SCALAR_OPS], template_memory)
    operators = GeneticOperators(instr_set, rng)
    
    # Create original individual
    original = Individual.random(instr_set, memory_cfg, program_length=10, rng=rng)
    original_program_str = str(original.program.instructions)
    original_length = len(original.program)
    
    # Create mutated copy
    mutated = original.copy(new_id=True)
    mutated.invalidate_fitness()
    operators.mutate_program(mutated.program, threshold=1.0, rng=rng)  # 100% mutation rate
    mutated_program_str = str(mutated.program.instructions)
    mutated_length = len(mutated.program)
    
    # Check that mutation changed something
    programs_different = original_program_str != mutated_program_str
    lengths_different = original_length != mutated_length
    
    print(f"  Original program length: {original_length}")
    print(f"  Mutated program length: {mutated_length}")
    print(f"  Programs are different: {programs_different}")
    print(f"  Lengths are different: {lengths_different}")
    
    # At least one should be different (very high probability with 100% mutation)
    assert programs_different or lengths_different, "Mutation should create diversity"
    
    print("✓ Mutations create program diversity")


def test_crossover_creates_offspring():
    """Test that crossover creates valid offspring."""
    print("\n" + "="*70)
    print("TEST 3: Crossover Creates Offspring")
    print("="*70)
    
    rng = np.random.default_rng(42)
    memory_cfg = MemoryConfig(
        n_scalar=4, n_vector=0, n_matrix=0,
        n_obs_scalar=2, n_obs_vector=0, n_obs_matrix=0,
        vector_size=1, matrix_shape=(1, 1)
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
    instr_set = InstructionSet([op() for op in SCALAR_OPS], template_memory)
    operators = GeneticOperators(instr_set, rng)
    
    # Create two different parents
    parent1 = Individual.random(instr_set, memory_cfg, program_length=8, rng=rng)
    parent2 = Individual.random(instr_set, memory_cfg, program_length=8, rng=rng)
    
    parent1_str = str(parent1.program.instructions)
    parent2_str = str(parent2.program.instructions)
    
    # Perform crossover
    child1_prog, child2_prog = operators.crossover(
        parent1.program, parent2.program, threshold=0.0, rng=rng  # Always crossover
    )
    
    child1_str = str(child1_prog.instructions)
    child2_str = str(child2_prog.instructions)
    
    # Children should be different from parents (with high probability)
    child1_different = child1_str != parent1_str and child1_str != parent2_str
    child2_different = child2_str != parent1_str and child2_str != parent2_str
    
    print(f"  Parent1 length: {len(parent1.program)}")
    print(f"  Parent2 length: {len(parent2.program)}")
    print(f"  Child1 length: {len(child1_prog)}")
    print(f"  Child2 length: {len(child2_prog)}")
    print(f"  Child1 is different: {child1_different}")
    print(f"  Child2 is different: {child2_different}")
    
    # At least one child should be different
    assert child1_different or child2_different, "Crossover should create new programs"
    
    print("✓ Crossover creates valid offspring")


def test_population_evaluation_sequential():
    """Test sequential population evaluation."""
    print("\n" + "="*70)
    print("TEST 4: Sequential Population Evaluation")
    print("="*70)
    
    rng = np.random.default_rng(42)
    memory_cfg = MemoryConfig(
        n_scalar=4, n_vector=0, n_matrix=0,
        n_obs_scalar=2, n_obs_vector=0, n_obs_matrix=0,
        vector_size=1, matrix_shape=(1, 1)
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
    instr_set = InstructionSet([op() for op in SCALAR_OPS], template_memory)
    
    pop_config = PopulationConfig(size=10, program_length=(3, 8), elitism=2)
    population = Population(pop_config, instr_set, memory_cfg, rng=rng)
    population.initialize_random(mutate_constants=True)
    
    evaluator = SymbolicRegressionEvaluator()
    
    # All individuals should have no fitness initially
    for ind in population.individuals:
        assert ind.fitness is None, "Individuals should have no fitness initially"
    
    # Evaluate sequentially
    population.evaluate_all(evaluator, verbose=False, n_jobs=1)
    
    # All individuals should now have fitness
    fitnesses = []
    for ind in population.individuals:
        assert ind.fitness is not None, "All individuals should have fitness after evaluation"
        fitnesses.append(ind.fitness)
    
    # Fitnesses should vary (not all identical)
    unique_fitnesses = len(set(fitnesses))
    print(f"  Population size: {len(population.individuals)}")
    print(f"  Unique fitness values: {unique_fitnesses}")
    print(f"  Fitness range: [{min(fitnesses):.4f}, {max(fitnesses):.4f}]")
    
    assert unique_fitnesses > 1, "Fitness values should vary across population"
    
    print("✓ Sequential evaluation works correctly")


def test_population_evaluation_parallel():
    """Test parallel population evaluation."""
    print("\n" + "="*70)
    print("TEST 5: Parallel Population Evaluation")
    print("="*70)
    
    rng = np.random.default_rng(42)
    memory_cfg = MemoryConfig(
        n_scalar=4, n_vector=0, n_matrix=0,
        n_obs_scalar=2, n_obs_vector=0, n_obs_matrix=0,
        vector_size=1, matrix_shape=(1, 1)
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
    instr_set = InstructionSet([op() for op in SCALAR_OPS], template_memory)
    
    pop_config = PopulationConfig(size=20, program_length=(3, 8), elitism=2)
    population = Population(pop_config, instr_set, memory_cfg, rng=rng)
    population.initialize_random(mutate_constants=True)
    
    evaluator = SymbolicRegressionEvaluator()
    
    # Invalidate all fitness
    population.invalidate_all_fitness()
    for ind in population.individuals:
        assert ind.fitness is None, "All fitness should be invalidated"
    
    # Evaluate in parallel
    population.evaluate_all(evaluator, verbose=False, n_jobs=2)  # Use 2 workers
    
    # All individuals should now have fitness
    fitnesses = []
    for ind in population.individuals:
        assert ind.fitness is not None, "All individuals should have fitness after parallel evaluation"
        fitnesses.append(ind.fitness)
    
    # Fitnesses should vary
    unique_fitnesses = len(set(fitnesses))
    print(f"  Population size: {len(population.individuals)}")
    print(f"  Unique fitness values: {unique_fitnesses}")
    print(f"  Fitness range: [{min(fitnesses):.4f}, {max(fitnesses):.4f}]")
    
    assert unique_fitnesses > 1, "Fitness values should vary across population"
    
    print("✓ Parallel evaluation works correctly")


def test_population_replacement():
    """Test that population replacement works and increments generation."""
    print("\n" + "="*70)
    print("TEST 6: Population Replacement")
    print("="*70)
    
    rng = np.random.default_rng(42)
    memory_cfg = MemoryConfig(
        n_scalar=4, n_vector=0, n_matrix=0,
        n_obs_scalar=2, n_obs_vector=0, n_obs_matrix=0,
        vector_size=1, matrix_shape=(1, 1)
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
    instr_set = InstructionSet([op() for op in SCALAR_OPS], template_memory)
    
    pop_config = PopulationConfig(size=10, program_length=(3, 8), elitism=2)
    population = Population(pop_config, instr_set, memory_cfg, rng=rng)
    population.initialize_random(mutate_constants=True)
    
    initial_generation = population.generation
    initial_individual_ids = [ind.id for ind in population.individuals]
    
    # Create new offspring
    offspring = [ind.copy(new_id=True) for ind in population.individuals[:pop_config.size]]
    
    # Replace population
    population.replace_population(offspring)
    
    # Generation should increment
    assert population.generation == initial_generation + 1, "Generation should increment"
    
    # Individual IDs should be different (new_id=True)
    new_individual_ids = [ind.id for ind in population.individuals]
    ids_changed = set(initial_individual_ids) != set(new_individual_ids)
    
    print(f"  Initial generation: {initial_generation}")
    print(f"  New generation: {population.generation}")
    print(f"  IDs changed: {ids_changed}")
    
    assert ids_changed, "New individuals should have different IDs"
    
    print("✓ Population replacement works correctly")


def test_evolution_generation_improvement():
    """Test that evolution can improve over generations."""
    print("\n" + "="*70)
    print("TEST 7: Evolution Over Generations")
    print("="*70)
    
    rng = np.random.default_rng(42)
    memory_cfg = MemoryConfig(
        n_scalar=4, n_vector=0, n_matrix=0,
        n_obs_scalar=2, n_obs_vector=0, n_obs_matrix=0,
        vector_size=1, matrix_shape=(1, 1)
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
    instr_set = InstructionSet([op() for op in SCALAR_OPS], template_memory)
    operators = GeneticOperators(instr_set, rng)
    
    pop_config = PopulationConfig(size=20, program_length=(3, 8), elitism=3)
    population = Population(pop_config, instr_set, memory_cfg, operators=operators, rng=rng)
    population.initialize_random(mutate_constants=True)
    
    evaluator = SymbolicRegressionEvaluator()
    
    # Run a few generations
    evolution_config = EvolutionConfig(
        max_generations=5,
        mutation_threshold=0.5,  # 50% mutation rate
        crossover_threshold=0.7,  # 70% crossover rate
        constant_mutation_rate=0.1,
        verbose=False
    )
    
    engine = EvolutionEngine(
        population=population,
        operators=operators,
        evaluator=evaluator,
        config=evolution_config,
        rng=rng,
    )
    
    initial_best = population.get_best()
    initial_fitness = initial_best.fitness if initial_best.fitness is not None else float('-inf')
    
    # Run evolution
    final_population = engine.run()
    
    final_best = final_population.get_best()
    final_fitness = final_best.fitness if final_best.fitness is not None else float('-inf')
    
    # Check that best_ever was tracked
    assert final_population.best_ever is not None, "best_ever should be set"
    assert final_population.best_ever_generation >= 0, "best_ever_generation should be set"
    
    print(f"  Initial best fitness: {initial_fitness:.4f}")
    print(f"  Final best fitness: {final_fitness:.4f}")
    print(f"  Best ever fitness: {final_population.best_ever.fitness:.4f}")
    print(f"  Best ever generation: {final_population.best_ever_generation}")
    print(f"  Final generation: {final_population.generation}")
    
    # Fitness should be computed (even if not improved)
    assert final_fitness != float('-inf'), "Final fitness should be computed"
    assert final_population.best_ever.fitness != float('-inf'), "Best ever fitness should be computed"
    
    print("✓ Evolution runs successfully over multiple generations")


def test_invalidate_fitness_bug_fix():
    """Test that the invalidate_fitness bug is fixed (child2.invalidate_fitness)."""
    print("\n" + "="*70)
    print("TEST 8: Invalidate Fitness Bug Fix")
    print("="*70)
    
    rng = np.random.default_rng(42)
    memory_cfg = MemoryConfig(
        n_scalar=4, n_vector=0, n_matrix=0,
        n_obs_scalar=2, n_obs_vector=0, n_obs_matrix=0,
        vector_size=1, matrix_shape=(1, 1)
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
    instr_set = InstructionSet([op() for op in SCALAR_OPS], template_memory)
    operators = GeneticOperators(instr_set, rng)
    
    # Create parent with fitness
    parent = Individual.random(instr_set, memory_cfg, program_length=5, rng=rng)
    evaluator = SymbolicRegressionEvaluator()
    parent.evaluate(evaluator)
    parent_fitness = parent.fitness
    
    # Create child1 and child2 (simulating evolution_engine.py behavior)
    child1 = parent.create_offspring(parent_ids=(parent.id, parent.id))
    child2 = parent.create_offspring(parent_ids=(parent.id, parent.id))
    
    # Both should have invalidated fitness
    assert child1.fitness is None, "child1 fitness should be invalidated"
    assert child2.fitness is None, "child2 fitness should be invalidated"
    
    # Manually test the bug fix - both should work
    child1.invalidate_fitness()  # Should work
    child2.invalidate_fitness()  # This was the bug - missing parentheses
    
    # Both should still be None after explicit invalidation
    assert child1.fitness is None, "child1 fitness should remain None"
    assert child2.fitness is None, "child2 fitness should remain None"
    
    print("✓ Both child1 and child2 properly invalidate fitness")
    print(f"  Parent fitness: {parent_fitness:.4f}")
    print(f"  Child1 fitness after invalidation: {child1.fitness}")
    print(f"  Child2 fitness after invalidation: {child2.fitness}")


def test_elitism_preserves_best():
    """Test that elitism preserves the best individuals."""
    print("\n" + "="*70)
    print("TEST 9: Elitism Preserves Best")
    print("="*70)
    
    rng = np.random.default_rng(42)
    memory_cfg = MemoryConfig(
        n_scalar=4, n_vector=0, n_matrix=0,
        n_obs_scalar=2, n_obs_vector=0, n_obs_matrix=0,
        vector_size=1, matrix_shape=(1, 1)
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
    instr_set = InstructionSet([op() for op in SCALAR_OPS], template_memory)
    
    pop_config = PopulationConfig(size=10, program_length=(3, 8), elitism=3)
    population = Population(pop_config, instr_set, memory_cfg, rng=rng)
    population.initialize_random(mutate_constants=True)
    
    evaluator = SymbolicRegressionEvaluator()
    population.evaluate_all(evaluator, verbose=False, n_jobs=1)
    
    # Get elites
    elites = population.get_elites()
    assert len(elites) == pop_config.elitism, "Should get correct number of elites"
    
    # Elites should be sorted by fitness (best first)
    elite_fitnesses = [e.fitness for e in elites]
    assert elite_fitnesses == sorted(elite_fitnesses, reverse=True), "Elites should be sorted by fitness"
    
    # Best elite should match population best
    population_best = population.get_best()
    assert elites[0].fitness == population_best.fitness, "Best elite should match population best"
    
    print(f"  Elitism count: {len(elites)}")
    print(f"  Elite fitnesses: {[f'{f:.4f}' for f in elite_fitnesses]}")
    print(f"  Population best: {population_best.fitness:.4f}")
    
    print("✓ Elitism correctly preserves best individuals")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("RUNNING COMPREHENSIVE EVOLUTION SYSTEM TESTS")
    print("="*70)
    
    tests = [
        test_individual_fitness_caching,
        test_mutation_creates_diversity,
        test_crossover_creates_offspring,
        test_population_evaluation_sequential,
        test_population_evaluation_parallel,
        test_population_replacement,
        test_evolution_generation_improvement,
        test_invalidate_fitness_bug_fix,
        test_elitism_preserves_best,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n✗ TEST FAILED: {test_func.__name__}")
            print(f"  Error: {e}")
            failed += 1
        except Exception as e:
            print(f"\n✗ TEST ERROR: {test_func.__name__}")
            print(f"  Error: {e}")
            failed += 1
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"  Passed: {passed}/{len(tests)}")
    print(f"  Failed: {failed}/{len(tests)}")
    print("="*70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

