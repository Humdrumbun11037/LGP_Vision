"""Test evolutionary loop with verbose output showing all individuals and their fitness."""

import numpy as np
import os
import sys
from pathlib import Path
from scipy.special import expit

# Add flappy-bird-env submodule to path if it exists (for direct import without pip install)
_submodule_path = Path(__file__).parent / "flappy-bird-env"
if _submodule_path.exists() and str(_submodule_path) not in sys.path:
    sys.path.insert(0, str(_submodule_path))

# Only set dummy driver if we're not using human rendering
# Check if render_mode will be "human" - if so, don't use dummy driver
USE_HUMAN_RENDERING = True  # Set to False to use headless mode (faster)

if not USE_HUMAN_RENDERING:
    # Initialize pygame for headless mode (required for FlappyBird)
    os.environ['SDL_VIDEODRIVER'] = 'dummy'

try:
    import pygame
    pygame.init()
    if USE_HUMAN_RENDERING:
        print("Pygame initialized for human rendering (window will appear)")
    else:
        print("Pygame initialized for headless mode (no window)")
except Exception as e:
    print(f"Warning: Could not initialize pygame: {e}")

import flappy_bird_env  # noqa
import gymnasium as gym
from memory_system import MemoryConfig, MemoryBank, MemoryType
from instruction_set import InstructionSet
from operation import AUTOML_ALL_OPS, CV_ALL_OPS
from individual import Individual
from population import Population, PopulationConfig
from operators import GeneticOperators
from evaluator import FlappyBirdEvaluator, FlappyBirdEvaluatorConfig
from evolution_engine import EvolutionEngine, EvolutionConfig


def test_individual_output_full_episode(individual, evaluator, max_steps=50):
    """Run a full episode for an individual and capture all outputs.
    
    Returns:
        List of tuples: [(step, scalar0_value, normalized_value, action), ...]
        Or None if error occurred
    """
    try:
        # Create a separate test environment to avoid interfering with main evaluation
        test_env = gym.make(evaluator.env_id, render_mode=None)  # No rendering for testing
        
        # Reset environment
        test_obs, _ = test_env.reset(seed=42)  # Use fixed seed for consistency
        test_obs = np.asarray(test_obs, dtype=np.float32)
        
        # Create fresh memory copy
        memory = individual.memory.copy()
        
        outputs = []
        
        # Run for max_steps or until episode ends
        for step in range(max_steps):
            # Process observation
            processed_observations, obs_type = evaluator._process_observation(test_obs)
            
            # Load observation
            if obs_type == 'vector':
                memory.load_observation({'vector': processed_observations})
            else:  # obs_type == 'matrix'
                memory.load_observation({'matrix': processed_observations})
            
            # Execute program
            individual.get_effective_program(evaluator.output_registers).execute(memory)
            
            # Read output register
            scalar0_value = memory.read_scalar(evaluator.output_register)
            
            # Calculate action (same logic as in evaluator)
            normalized = expit(scalar0_value)  # sigmoid
            action = 1 if normalized >= 0.5 else 0
            
            # Store this step's output
            outputs.append((step, scalar0_value, normalized, action))
            
            # Take action and get next observation
            test_obs, reward, terminated, truncated, _ = test_env.step(action)
            test_obs = np.asarray(test_obs, dtype=np.float32)
            
            if terminated or truncated:
                break
        
        # Cleanup test environment
        test_env.close()
        return outputs
    except Exception as e:
        print(f"Error testing individual {individual.id}: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_population_details(population, generation):
    """Print detailed information about every individual in the population."""
    print("\n" + "="*80)
    print(f"GENERATION {generation} - ALL INDIVIDUALS")
    print("="*80)
    
    # Get evaluator for testing outputs
    evaluator = None
    if hasattr(population, 'evaluator'):
        evaluator = population.evaluator
    
    # Sort by fitness (best first)
    sorted_individuals = sorted(
        population.individuals,
        key=lambda ind: ind.fitness if ind.fitness is not None else float('-inf'),
        reverse=True
    )
    
    print(f"\nPopulation size: {len(population.individuals)}")
    print(f"{'Index':<6} {'ID':<6} {'Fitness':<12} {'Length':<8} {'Age':<6} {'Steps':<8} {'Eff'}")
    print("-" * 80)
    
    all_outputs = []  # Store all outputs for statistics
    
    for idx, individual in enumerate(sorted_individuals):
        fitness_str = f"{individual.fitness:.6f}" if individual.fitness is not None else "None"
        program_str = f"{len(individual.program)} instr"
        
        # Get effective length if output_registers are available
        effective_info = ""
        if hasattr(population, 'evaluator') and hasattr(population.evaluator, 'output_registers'):
            output_registers = population.evaluator.output_registers
            if output_registers:
                effective_length = individual.get_effective_length(output_registers)
                effective_info = f"{effective_length}"
        
        # Test individual output for full episode
        steps_str = "N/A"
        if evaluator is not None:
            episode_outputs = test_individual_output_full_episode(individual, evaluator, max_steps=50)
            if episode_outputs is not None:
                steps_str = f"{len(episode_outputs)}"
                all_outputs.append((individual.id, episode_outputs))
            else:
                steps_str = "ERROR"
        
        print(f"{idx:<6} {individual.id:<6} {fitness_str:<12} {program_str:<8} {individual.age:<6} {steps_str:<8} {effective_info}")
    
    # Print statistics
    fitnesses = [ind.fitness for ind in population.individuals if ind.fitness is not None]
    if fitnesses:
        print("\n" + "-" * 80)
        print(f"Fitness Statistics:")
        print(f"  Min:    {min(fitnesses):.6f}")
        print(f"  Max:    {max(fitnesses):.6f}")
        print(f"  Mean:   {np.mean(fitnesses):.6f}")
        print(f"  Std:    {np.std(fitnesses):.6f}")
        print(f"  Unique: {len(set(fitnesses))} unique values")
    
    # Print program length statistics
    lengths = [len(ind.program) for ind in population.individuals]
    print(f"\nProgram Length Statistics:")
    print(f"  Min:    {min(lengths)}")
    print(f"  Max:    {max(lengths)}")
    print(f"  Mean:   {np.mean(lengths):.2f}")
    print(f"  Std:    {np.std(lengths):.2f}")
    print(f"  Unique: {len(set(lengths))} unique lengths")
    
    # Print detailed output for each individual
    if all_outputs:
        print(f"\n" + "="*80)
        print(f"DETAILED OUTPUTS FOR EACH INDIVIDUAL")
        print("="*80)
        
        all_scalar_values = []
        all_actions = []
        action_counts = {0: 0, 1: 0}
        
        for individual_id, episode_outputs in all_outputs:
            print(f"\n--- Individual ID {individual_id} ---")
            print(f"Total steps: {len(episode_outputs)}")
            print(f"{'Step':<6} {'Scalar0':<12} {'Normalized':<12} {'Action':<8} {'ActionName'}")
            print("-" * 60)
            
            for step, scalar0, normalized, action in episode_outputs:
                action_name = "FLAP" if action == 1 else "NOOP"
                print(f"{step:<6} {scalar0:<12.6f} {normalized:<12.6f} {action:<8} {action_name}")
                all_scalar_values.append(scalar0)
                all_actions.append(action)
                action_counts[action] = action_counts.get(action, 0) + 1
        
        # Print aggregate statistics
        if all_scalar_values:
            print(f"\n" + "="*80)
            print(f"AGGREGATE OUTPUT STATISTICS (All Individuals, All Steps)")
            print("="*80)
            print(f"\nScalar 0 (Output Register) Statistics:")
            print(f"  Min:    {min(all_scalar_values):.6f}")
            print(f"  Max:    {max(all_scalar_values):.6f}")
            print(f"  Mean:   {np.mean(all_scalar_values):.6f}")
            print(f"  Std:    {np.std(all_scalar_values):.6f}")
            print(f"  Unique: {len(set(all_scalar_values))} unique values")
            
            print(f"\nAction Choice Distribution (All Steps):")
            total_actions = len(all_actions)
            for action, count in sorted(action_counts.items()):
                percentage = 100.0 * count / total_actions if total_actions > 0 else 0.0
                action_name = "FLAP" if action == 1 else "NOOP"
                print(f"  Action {action} ({action_name}): {count}/{total_actions} ({percentage:.1f}%)")
            
            if len(set(all_scalar_values)) == 1:
                print(f"\n⚠️  WARNING: All individuals produce the SAME output value at all steps!")
                print(f"   This suggests programs aren't creating diversity in behavior.")
            
            # Check if all actions are the same
            unique_actions = set(all_actions)
            if len(unique_actions) == 1:
                action_val = list(unique_actions)[0]
                action_name = "FLAP" if action_val == 1 else "NOOP"
                print(f"\n⚠️  WARNING: All individuals choose the SAME action ({action_name}) at all steps!")
                print(f"   This explains why fitness is identical - all programs behave the same.")
    
    print("="*80)


def print_individual_programs(population, generation, top_n=3):
    """Print the actual program code for top N individuals."""
    print(f"\n{'='*80}")
    print(f"GENERATION {generation} - TOP {top_n} PROGRAMS")
    print("="*80)
    
    sorted_individuals = sorted(
        population.individuals,
        key=lambda ind: ind.fitness if ind.fitness is not None else float('-inf'),
        reverse=True
    )
    
    output_registers = None
    if hasattr(population, 'evaluator') and hasattr(population.evaluator, 'output_registers'):
        output_registers = population.evaluator.output_registers
    
    for rank, individual in enumerate(sorted_individuals[:top_n], 1):
        print(f"\n--- Rank {rank}: Individual ID {individual.id} ---")
        print(f"Fitness: {individual.fitness:.6f}" if individual.fitness is not None else "Fitness: None")
        print(f"Program Length: {len(individual.program)}")
        
        if output_registers:
            effective_length = individual.get_effective_length(output_registers)
            effective_rate = effective_length / len(individual.program) if len(individual.program) > 0 else 0.0
            print(f"Effective Length: {effective_length} (rate: {effective_rate:.3f})")
        
        print("Program Instructions:")
        for i, instr in enumerate(individual.program.instructions):
            print(f"  {i:3d}: {instr}")
        
        if output_registers:
            effective_program = individual.get_effective_program(output_registers)
            print(f"\nEffective Program ({len(effective_program)} instructions):")
            for i, instr in enumerate(effective_program.instructions):
                print(f"  {i:3d}: {instr}")


def test_evolution_verbose():
    """Run evolutionary loop with 2 generations and print all individuals."""
    print("\n" + "="*80)
    print("EVOLUTIONARY LOOP TEST - 2 GENERATIONS")
    print("="*80)
    
    # Setup
    rng = np.random.default_rng(42)
    
    # Memory config for FlappyBird (matrix observations)
    memory_cfg = MemoryConfig(
        n_scalar=8,
        n_vector=8,
        n_matrix=8,
        n_obs_scalar=0,
        n_obs_vector=0,
        n_obs_matrix=1,  # Single matrix observation for quantized image
        vector_size=37,  # Matching quantization factor 0.05 -> 37x37
        matrix_shape=(37, 37)
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
    
    # Use AutoML + CV operations for image processing
    all_ops = AUTOML_ALL_OPS + CV_ALL_OPS
    instruction_set = InstructionSet([op() for op in all_ops], template_memory)
    operators = GeneticOperators(instruction_set, rng)
    
    # Small population for easy viewing
    pop_config = PopulationConfig(
        size=10,  # Small population so we can see all individuals
        program_length=(3, 8),
        elitism=2,
        max_program_length=50,
    )
    
    population = Population(
        pop_config,
        instruction_set,
        memory_cfg,
        operators=operators,
        rng=rng,
    )
    population.initialize_random(mutate_constants=True)
    
    # Create FlappyBird evaluator
    # Use "human" to see the game, "rgb_array" for headless (faster)
    render_mode = "human" if USE_HUMAN_RENDERING else "rgb_array"
    
    evaluator_config = FlappyBirdEvaluatorConfig(
        env_id="FlappyBird-v0",
        episodes=2,  # Reduced for faster testing
        max_steps=100,  # Reduced for faster testing
        output_register=0,
        render_mode=render_mode,
        rng_seed=42,
        patch_strategy="quantized",
        color_channel=2,  # Blue channel
        normalize=True,
        quantization_factor=0.05,
        output_registers=[(MemoryType.SCALAR, 0)],  # Explicitly set output_registers
        n_jobs=1,  # Sequential for verbose output
    )
    
    print(f"\nCreating FlappyBird evaluator with render_mode='{render_mode}'...")
    evaluator = FlappyBirdEvaluator(config=evaluator_config)
    print("✓ FlappyBird evaluator created successfully!")
    
    # Test that environment is working by resetting it
    print("\nTesting environment creation...")
    test_obs, test_info = evaluator.env.reset()
    print(f"✓ Environment reset successful! Observation shape: {test_obs.shape}")
    print(f"  Observation dtype: {test_obs.dtype}")
    print(f"  Observation range: [{test_obs.min():.1f}, {test_obs.max():.1f}]")
    
    if render_mode == "human":
        print("\n⚠️  NOTE: With render_mode='human', FlappyBird windows will appear during evaluation.")
        print("   This will be slower but you can watch the game!")
    else:
        print("\nℹ️  Using headless mode (render_mode='rgb_array') - no windows will appear.")
    
    # Store evaluator in population for access in print functions
    population.evaluator = evaluator
    
    print(f"\nFlappyBird Evaluator Configuration:")
    print(f"  Episodes per evaluation: {evaluator.episodes}")
    print(f"  Max steps per episode: {evaluator.max_steps}")
    print(f"  Output register: {evaluator.output_register}")
    print(f"  Output registers: {evaluator.output_registers}")
    print(f"  Patch strategy: {evaluator.patch_strategy}")
    print(f"  Quantization factor: {evaluator.quantization_factor}")
    
    print("\n" + "="*80)
    print("INITIAL POPULATION (Before Evaluation)")
    print("="*80)
    print(f"Population size: {len(population.individuals)}")
    print("All individuals have fitness = None initially")
    
    # Generation 0: Evaluate initial population
    print("\n" + "="*80)
    print("EVALUATING GENERATION 0")
    print("="*80)
    population.evaluate_all(evaluator, verbose=True, n_jobs=1)
    
    # Print all individuals after generation 0
    print_population_details(population, 0)
    print_individual_programs(population, 0, top_n=3)
    
    # Setup evolution engine
    evolution_config = EvolutionConfig(
        max_generations=2,
        mutation_threshold=0.5,  # 50% mutation rate
        crossover_threshold=0.7,  # 70% crossover rate
        constant_mutation_rate=0.1,
        verbose=False,  # We'll print our own verbose output
    )
    
    engine = EvolutionEngine(
        population=population,
        operators=operators,
        evaluator=evaluator,
        config=evolution_config,
        rng=rng,
    )
    
    # Generation 1: Run one generation
    print("\n" + "="*80)
    print("RUNNING GENERATION 1")
    print("="*80)
    
    # Manually run one generation to have control over printing
    # Evaluate current population
    population.evaluate_all(evaluator, verbose=True, n_jobs=1)
    
    # Track best-ever
    current_best = population.get_best()
    if current_best.fitness is not None:
        was_none = population.best_ever is None
        if was_none or current_best.fitness > population.best_ever.fitness:
            population.best_ever = current_best.copy(new_id=False)
            population.best_ever_generation = population.generation
            print(f"  → Updated best_ever (fitness: {current_best.fitness:.6f}, gen: {population.generation})")
    
    # Record statistics
    population.record_statistics()
    engine._record_best_agent(population.generation)
    
    # Print all individuals after generation 1 (before replacement)
    print_population_details(population, 1)
    print_individual_programs(population, 1, top_n=3)
    
    # Create next generation
    print("\n" + "="*80)
    print("CREATING GENERATION 2 (Offspring)")
    print("="*80)
    
    elites = population.get_elites()
    print(f"Selected {len(elites)} elites:")
    for elite in elites:
        print(f"  ID {elite.id}: fitness={elite.fitness:.6f}, length={len(elite.program)}")
    
    offspring = elites.copy()
    
    print(f"\nCreating {pop_config.size - len(elites)} new offspring...")
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
        
        # Mutate constants if needed
        if evolution_config.constant_mutation_rate > 0 and rng.random() < evolution_config.constant_mutation_rate:
            operators.mutate_constants(child1.memory, rng)
            operators.mutate_constants(child2.memory, rng)
        
        child1.invalidate_fitness()
        child2.invalidate_fitness()
        offspring.append(child1)
        offspring.append(child2)
    
    offspring = offspring[:population.config.size]
    
    print(f"Created {len(offspring)} offspring (including {len(elites)} elites)")
    print(f"Offspring fitness status: {sum(1 for ind in offspring if ind.fitness is None)} have None, "
          f"{sum(1 for ind in offspring if ind.fitness is not None)} have fitness")
    
    # Replace population
    population.replace_population(offspring)
    print(f"Generation after replacement: {population.generation}")
    
    # Generation 2: Evaluate new population
    print("\n" + "="*80)
    print("EVALUATING GENERATION 2")
    print("="*80)
    population.evaluate_all(evaluator, verbose=True, n_jobs=1)
    
    # Track best-ever
    current_best = population.get_best()
    if current_best.fitness is not None:
        was_none = population.best_ever is None
        if was_none or current_best.fitness > population.best_ever.fitness:
            population.best_ever = current_best.copy(new_id=False)
            population.best_ever_generation = population.generation
            print(f"  → Updated best_ever (fitness: {current_best.fitness:.6f}, gen: {population.generation})")
    
    # Record statistics
    population.record_statistics()
    engine._record_best_agent(population.generation)
    
    # Print all individuals after generation 2
    print_population_details(population, 2)
    print_individual_programs(population, 2, top_n=3)
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Total generations run: {population.generation + 1}")
    if population.best_ever is not None:
        print(f"Best ever fitness: {population.best_ever.fitness:.6f} (Generation {population.best_ever_generation})")
        print(f"Best ever program length: {len(population.best_ever.program)}")
        
        # Get output_registers from evaluator
        output_registers = None
        if hasattr(evaluator, 'output_registers'):
            output_registers = evaluator.output_registers
        if output_registers:
            effective_length = population.best_ever.get_effective_length(output_registers)
            print(f"Best ever effective length: {effective_length}")
    
    print("\n" + "="*80)
    print("EVOLUTION TEST COMPLETE")
    print("="*80)
    
    # Cleanup
    evaluator.close()
    
    return population


if __name__ == "__main__":
    try:
        population = test_evolution_verbose()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n\nError during test: {e}")
        import traceback
        traceback.print_exc()

