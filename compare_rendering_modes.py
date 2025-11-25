"""Compare headless vs human rendering modes - speed and result verification."""

import time
import numpy as np
import os
import sys
from pathlib import Path
from scipy.special import expit

# Add flappy-bird-env submodule to path if it exists (for direct import without pip install)
_submodule_path = Path(__file__).parent / "flappy-bird-env"
if _submodule_path.exists() and str(_submodule_path) not in sys.path:
    sys.path.insert(0, str(_submodule_path))

# Test configuration
POPULATION_SIZE = 5  # Small for faster comparison
EPISODES = 1
MAX_STEPS = 20  # Short episodes for speed comparison

def run_test_with_mode(use_human_rendering):
    """Run the evolution test with specified rendering mode."""
    
    # Set up environment
    if not use_human_rendering:
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
    else:
        # Remove dummy driver if it was set
        os.environ.pop('SDL_VIDEODRIVER', None)
    
    try:
        import pygame
        pygame.init()
    except Exception:
        pass
    
    import flappy_bird_env  # noqa
    import gymnasium as gym
    from memory_system import MemoryConfig, MemoryBank, MemoryType
    from instruction_set import InstructionSet
    from operation import AUTOML_ALL_OPS, CV_ALL_OPS
    from individual import Individual
    from population import Population, PopulationConfig
    from operators import GeneticOperators
    from evaluator import FlappyBirdEvaluator, FlappyBirdEvaluatorConfig
    
    # Setup
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    
    memory_cfg = MemoryConfig(
        n_scalar=8,
        n_vector=8,
        n_matrix=8,
        n_obs_scalar=0,
        n_obs_vector=0,
        n_obs_matrix=1,
        vector_size=37,
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
    
    all_ops = AUTOML_ALL_OPS + CV_ALL_OPS
    instruction_set = InstructionSet([op() for op in all_ops], template_memory)
    operators = GeneticOperators(instruction_set, rng)
    
    pop_config = PopulationConfig(
        size=POPULATION_SIZE,
        program_length=(3, 6),
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
    
    # Create evaluator
    render_mode = "human" if use_human_rendering else "rgb_array"
    
    evaluator_config = FlappyBirdEvaluatorConfig(
        env_id="FlappyBird-v0",
        episodes=EPISODES,
        max_steps=MAX_STEPS,
        output_register=0,
        render_mode=render_mode,
        rng_seed=42,
        patch_strategy="quantized",
        color_channel=2,
        normalize=True,
        quantization_factor=0.05,
        output_registers=[(MemoryType.SCALAR, 0)],
        n_jobs=1,
    )
    
    evaluator = FlappyBirdEvaluator(config=evaluator_config)
    
    # Evaluate population
    start_time = time.time()
    population.evaluate_all(evaluator, verbose=False, n_jobs=1)
    elapsed_time = time.time() - start_time
    
    # Collect results
    results = {
        'fitnesses': [ind.fitness for ind in population.individuals if ind.fitness is not None],
        'individual_ids': [ind.id for ind in population.individuals],
        'program_lengths': [len(ind.program) for ind in population.individuals],
        'elapsed_time': elapsed_time,
    }
    
    evaluator.close()
    
    return results


def compare_modes():
    """Compare headless vs human rendering modes."""
    print("\n" + "="*80)
    print("RENDERING MODE COMPARISON TEST")
    print("="*80)
    print(f"\nTest Configuration:")
    print(f"  Population size: {POPULATION_SIZE}")
    print(f"  Episodes per evaluation: {EPISODES}")
    print(f"  Max steps per episode: {MAX_STEPS}")
    print(f"  Random seed: 42 (fixed for reproducibility)")
    
    # Run headless mode
    print("\n" + "="*80)
    print("TEST 1: HEADLESS MODE (rgb_array)")
    print("="*80)
    headless_start = time.time()
    headless_results = run_test_with_mode(use_human_rendering=False)
    headless_total = time.time() - headless_start
    
    print(f"\n✓ Headless mode completed")
    print(f"  Evaluation time: {headless_results['elapsed_time']:.3f} seconds")
    print(f"  Total time: {headless_total:.3f} seconds")
    print(f"  Fitnesses: {[f'{f:.6f}' for f in headless_results['fitnesses']]}")
    
    # Run human rendering mode
    print("\n" + "="*80)
    print("TEST 2: HUMAN RENDERING MODE (human)")
    print("="*80)
    human_start = time.time()
    human_results = run_test_with_mode(use_human_rendering=True)
    human_total = time.time() - human_start
    
    print(f"\n✓ Human rendering mode completed")
    print(f"  Evaluation time: {human_results['elapsed_time']:.3f} seconds")
    print(f"  Total time: {human_total:.3f} seconds")
    print(f"  Fitnesses: {[f'{f:.6f}' for f in human_results['fitnesses']]}")
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    # Time comparison
    speedup = human_total / headless_total
    print(f"\n⏱️  SPEED COMPARISON:")
    print(f"  Headless mode:  {headless_total:.3f} seconds")
    print(f"  Human mode:     {human_total:.3f} seconds")
    print(f"  Speedup:        {speedup:.2f}x faster with headless mode")
    print(f"  Time saved:     {human_total - headless_total:.3f} seconds ({100*(1-1/speedup):.1f}% faster)")
    
    # Result verification
    print(f"\n✅ RESULT VERIFICATION:")
    
    # Compare fitnesses (should be identical with same seed)
    fitness_match = np.allclose(
        sorted(headless_results['fitnesses']),
        sorted(human_results['fitnesses']),
        rtol=1e-6
    )
    
    if fitness_match:
        print(f"  ✓ Fitness values are IDENTICAL (within tolerance)")
    else:
        print(f"  ✗ Fitness values DIFFER!")
        print(f"    Headless: {sorted(headless_results['fitnesses'])}")
        print(f"    Human:    {sorted(human_results['fitnesses'])}")
    
    # Compare program lengths (should be identical - same initialization)
    lengths_match = headless_results['program_lengths'] == human_results['program_lengths']
    if lengths_match:
        print(f"  ✓ Program lengths are IDENTICAL")
    else:
        print(f"  ✗ Program lengths DIFFER!")
        print(f"    Headless: {headless_results['program_lengths']}")
        print(f"    Human:    {human_results['program_lengths']}")
    
    # Compare individual IDs (should be identical - same initialization)
    ids_match = headless_results['individual_ids'] == human_results['individual_ids']
    if ids_match:
        print(f"  ✓ Individual IDs are IDENTICAL")
    else:
        print(f"  ✗ Individual IDs DIFFER!")
    
    # Detailed fitness comparison
    print(f"\n📊 DETAILED FITNESS COMPARISON:")
    print(f"{'ID':<6} {'Headless':<12} {'Human':<12} {'Match':<8}")
    print("-" * 40)
    
    # Sort by ID for comparison
    headless_dict = dict(zip(headless_results['individual_ids'], headless_results['fitnesses']))
    human_dict = dict(zip(human_results['individual_ids'], human_results['fitnesses']))
    
    all_ids = sorted(set(headless_results['individual_ids'] + human_results['individual_ids']))
    all_match = True
    
    for ind_id in all_ids:
        hless_fit = headless_dict.get(ind_id, None)
        human_fit = human_dict.get(ind_id, None)
        
        if hless_fit is not None and human_fit is not None:
            match = abs(hless_fit - human_fit) < 1e-6
            match_str = "✓" if match else "✗"
            if not match:
                all_match = False
            print(f"{ind_id:<6} {hless_fit:<12.6f} {human_fit:<12.6f} {match_str:<8}")
        else:
            print(f"{ind_id:<6} {'MISSING':<12} {'MISSING':<12} {'✗':<8}")
            all_match = False
    
    # Final verdict
    print("\n" + "="*80)
    if fitness_match and lengths_match and ids_match and all_match:
        print("✅ VERDICT: Results are IDENTICAL - Rendering mode does not affect results!")
    else:
        print("⚠️  VERDICT: Some differences detected - check details above")
    
    print(f"\n💡 RECOMMENDATION:")
    if speedup > 1.1:
        print(f"   Use HEADLESS mode for {speedup:.1f}x speedup during evolution")
        print(f"   Use HUMAN mode only when you want to watch specific individuals")
    else:
        print(f"   Speed difference is minimal - use either mode")
    
    print("="*80)
    
    return {
        'headless_time': headless_total,
        'human_time': human_total,
        'speedup': speedup,
        'results_match': fitness_match and lengths_match and ids_match and all_match
    }


if __name__ == "__main__":
    try:
        results = compare_modes()
        exit(0 if results['results_match'] else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        exit(1)
    except Exception as e:
        print(f"\n\nError during comparison: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

