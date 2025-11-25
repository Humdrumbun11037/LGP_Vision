"""Diagnostic script to check if evolution is working correctly."""

import numpy as np
from memory_system import MemoryConfig, MemoryBank
from instruction_set import InstructionSet
from operation import AUTOML_ALL_OPS, CV_ALL_OPS
from individual import Individual
from population import Population, PopulationConfig
from operators import GeneticOperators
from evaluator import FlappyBirdEvaluatorConfig, FlappyBirdEvaluator
from evolution_engine import EvolutionEngine, EvolutionConfig


def diagnose_evolution():
    """Run diagnostics on the evolution system."""
    print("\n" + "="*70)
    print("EVOLUTION SYSTEM DIAGNOSTICS")
    print("="*70)
    
    rng = np.random.default_rng(5)
    
    # Setup (matching your notebook config)
    memory_cfg = MemoryConfig(
        n_scalar=8,
        n_vector=8,
        n_matrix=8,
        n_obs_scalar=0,
        n_obs_vector=0,
        n_obs_matrix=1,
        vector_size=37,
        matrix_shape=(37, 37),
    )
    
    all_ops = AUTOML_ALL_OPS + CV_ALL_OPS
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
    instruction_set = InstructionSet([op() for op in all_ops], template_memory)
    operators = GeneticOperators(instruction_set, rng)
    
    # Create evaluator
    # Note: FlappyBird requires render_mode="rgb_array" even when not displaying
    # because pygame needs to be initialized
    evaluator_config = FlappyBirdEvaluatorConfig(
        env_id="FlappyBird-v0",
        episodes=2,  # Reduced for faster testing
        max_steps=100,  # Reduced for faster testing
        output_register=0,
        render_mode="human",  # Required for pygame initialization
        rng_seed=5,
        patch_strategy="quantized",
        color_channel=1,
        normalize=True,
        quantization_factor=0.05,
        output_registers=[(MemoryType.SCALAR, 0)],
        n_jobs=1,  # Sequential for diagnostics
    )
    evaluator = FlappyBirdEvaluator(config=evaluator_config)
    
    # Small population
    pop_config = PopulationConfig(
        size=10,  # Small for faster testing
        program_length=(1, 5),
        elitism=9,
        max_program_length=250,
    )
    
    population = Population(
        pop_config,
        instruction_set,
        memory_cfg,
        operators=operators,
        rng=rng,
    )
    population.initialize_random(mutate_constants=True)
    
    print("\n1. INITIAL POPULATION")
    print("-" * 70)
    print(f"  Population size: {len(population.individuals)}")
    print(f"  Generation: {population.generation}")
    
    # Check initial diversity
    lengths = [len(ind.program) for ind in population.individuals]
    print(f"  Program lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")
    unique_lengths = len(set(lengths))
    print(f"  Unique program lengths: {unique_lengths}")
    
    # Evaluate initial population
    print("\n2. EVALUATING INITIAL POPULATION")
    print("-" * 70)
    population.evaluate_all(evaluator, verbose=True, n_jobs=1)
    
    initial_fitnesses = [ind.fitness for ind in population.individuals if ind.fitness is not None]
    if initial_fitnesses:
        print(f"\n  Fitness statistics:")
        print(f"    Min: {min(initial_fitnesses):.4f}")
        print(f"    Max: {max(initial_fitnesses):.4f}")
        print(f"    Mean: {np.mean(initial_fitnesses):.4f}")
        print(f"    Std: {np.std(initial_fitnesses):.4f}")
        print(f"    Unique values: {len(set(initial_fitnesses))}")
        
        initial_best = population.get_best()
        print(f"\n  Best individual:")
        print(f"    Fitness: {initial_best.fitness:.4f}")
        print(f"    Program length: {len(initial_best.program)}")
    else:
        print("  ERROR: No individuals have fitness!")
        return False
    
    # Run one generation
    print("\n3. RUNNING ONE GENERATION")
    print("-" * 70)
    
    evolution_config = EvolutionConfig(
        max_generations=1,
        mutation_threshold=0.9,
        crossover_threshold=0.9,
        constant_mutation_rate=0.1,
        verbose=True,
    )
    
    engine = EvolutionEngine(
        population=population,
        operators=operators,
        evaluator=evaluator,
        config=evolution_config,
        rng=rng,
    )
    
    # Manually run one generation to see what happens
    population.evaluate_all(evaluator, verbose=True, n_jobs=1)
    
    # Get elites
    elites = population.get_elites()
    print(f"\n  Elites selected: {len(elites)}")
    if elites:
        elite_fitnesses = [e.fitness for e in elites]
        print(f"  Elite fitnesses: {[f'{f:.4f}' for f in elite_fitnesses]}")
    
    # Create offspring
    print("\n  Creating offspring...")
    offspring = elites.copy()
    
    while len(offspring) < population.config.size:
        parent1, parent2 = population.tournament_selection(3, num_winners=2)
        child_program_1, child_program_2 = operators.crossover(
            parent1.program, parent2.program, evolution_config.crossover_threshold, rng
        )
        
        # Mutate children
        operators.mutate_program(
            child_program_1,
            evolution_config.mutation_threshold,
            rng,
            max_length=population.config.max_program_length,
        )
        operators.mutate_program(
            child_program_2,
            evolution_config.mutation_threshold,
            rng,
            max_length=population.config.max_program_length,
        )
        
        # Create child individuals
        child1 = parent1.create_offspring(parent_ids=(parent1.id, parent2.id))
        child1.program = child_program_1
        child2 = parent2.create_offspring(parent_ids=(parent1.id, parent2.id))
        child2.program = child_program_2
        
        # Check if fitness is invalidated
        if child1.fitness is not None:
            print(f"    WARNING: child1 still has fitness {child1.fitness:.4f} after create_offspring!")
        if child2.fitness is not None:
            print(f"    WARNING: child2 still has fitness {child2.fitness:.4f} after create_offspring!")
        
        # Explicitly invalidate (testing the bug fix)
        child1.invalidate_fitness()
        child2.invalidate_fitness()  # This was the bug - should have parentheses
        
        if child1.fitness is not None:
            print(f"    ERROR: child1 fitness not invalidated! Value: {child1.fitness:.4f}")
        if child2.fitness is not None:
            print(f"    ERROR: child2 fitness not invalidated! Value: {child2.fitness:.4f}")
        
        # Mutate constants if needed
        if evolution_config.constant_mutation_rate > 0 and rng.random() < evolution_config.constant_mutation_rate:
            operators.mutate_constants(child1.memory, rng)
            operators.mutate_constants(child2.memory, rng)
        
        offspring.append(child1)
        offspring.append(child2)
    
    offspring = offspring[:population.config.size]
    
    # Check offspring diversity
    offspring_lengths = [len(ind.program) for ind in offspring]
    print(f"\n  Offspring program lengths: min={min(offspring_lengths)}, max={max(offspring_lengths)}, mean={np.mean(offspring_lengths):.1f}")
    unique_offspring_lengths = len(set(offspring_lengths))
    print(f"  Unique offspring lengths: {unique_offspring_lengths}")
    
    # Replace population
    population.replace_population(offspring)
    print(f"\n  Generation after replacement: {population.generation}")
    
    # Evaluate new population
    print("\n4. EVALUATING NEW GENERATION")
    print("-" * 70)
    population.evaluate_all(evaluator, verbose=True, n_jobs=1)
    
    new_fitnesses = [ind.fitness for ind in population.individuals if ind.fitness is not None]
    if new_fitnesses:
        print(f"\n  Fitness statistics:")
        print(f"    Min: {min(new_fitnesses):.4f}")
        print(f"    Max: {max(new_fitnesses):.4f}")
        print(f"    Mean: {np.mean(new_fitnesses):.4f}")
        print(f"    Std: {np.std(new_fitnesses):.4f}")
        print(f"    Unique values: {len(set(new_fitnesses))}")
        
        new_best = population.get_best()
        print(f"\n  Best individual:")
        print(f"    Fitness: {new_best.fitness:.4f}")
        print(f"    Program length: {len(new_best.program)}")
        
        # Compare with initial
        print(f"\n  Comparison:")
        print(f"    Initial best: {initial_best.fitness:.4f}")
        print(f"    New best: {new_best.fitness:.4f}")
        print(f"    Improvement: {new_best.fitness - initial_best.fitness:.4f}")
    else:
        print("  ERROR: No individuals have fitness after evaluation!")
        return False
    
    # Check best_ever tracking
    print("\n5. BEST_EVER TRACKING")
    print("-" * 70)
    if population.best_ever is not None:
        print(f"  Best ever fitness: {population.best_ever.fitness:.4f}")
        print(f"  Best ever generation: {population.best_ever_generation}")
    else:
        print("  WARNING: best_ever is None!")
    
    evaluator.close()
    
    print("\n" + "="*70)
    print("DIAGNOSTICS COMPLETE")
    print("="*70)
    
    return True


if __name__ == "__main__":
    try:
        import flappy_bird_env  # noqa
    except ImportError:
        print("ERROR: flappy_bird_env not available. Skipping FlappyBird tests.")
        print("Run the simpler test_evolution.py instead.")
        exit(1)
    
    from memory_system import MemoryType
    
    # Initialize pygame for headless mode (required for FlappyBird)
    import os
    os.environ['SDL_VIDEODRIVER'] = 'dummy'  # Use dummy video driver for headless
    
    try:
        import pygame
        pygame.init()
    except Exception as e:
        print(f"Warning: Could not initialize pygame: {e}")
        print("This may cause issues with FlappyBird environment.")
    
    success = diagnose_evolution()
    exit(0 if success else 1)

