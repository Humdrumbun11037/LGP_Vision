#!/usr/bin/env python3
"""Run FlappyBird evolution and generate fitness chart."""

import sys
import argparse
from pathlib import Path
from typing import Optional
import flappy_bird_env  # noqa
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import time

import gymnasium as gym
from memory_system import MemoryConfig
from instruction_set import InstructionSet
from operation import AUTOML_ALL_OPS, AUTOML_NO_RANDOM_OPS, CV_ALL_OPS, FLAPPYBIRD_MINIMAL_OPS, MINIMAL_SCALAR_OPS, FEATURE_VECTOR_OPS
from population import Population, PopulationConfig
from operators import GeneticOperators
from evaluator import FlappyBirdEvaluator, FlappyBirdEvaluatorConfig
from evolution_engine import EvolutionEngine, EvolutionConfig
from experiment_manager import ExperimentManager
from modes import MODESTracker

# Import config loader
from config_loader import (
    load_config,
    create_memory_config,
    create_evaluator_config,
    create_population_config,
    create_evolution_config,
    get_operations_config,
    get_experiment_config,
)


def main(config_path: str = "config.yaml", random_seed: Optional[int] = None):
    """Run FlappyBird evolution.
    
    Args:
        config_path: Path to YAML configuration file
        random_seed: Random seed (required, overrides config)
    """
    if random_seed is None:
        raise ValueError("Random seed must be provided via --seed argument")
    
    # Load configuration from YAML
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    # Override config with command line seed
    config['random_seed'] = random_seed
    rng = np.random.default_rng(random_seed)
    print(f"Using random seed: {random_seed}")

    # Create experiment manager (creates directory structure, saves config copy)
    exp_config = get_experiment_config(config)
    
    # Always append seed to experiment name
    exp_name = exp_config.get('name', '')
    if exp_name:
        exp_config['name'] = f"{exp_name}_seed{random_seed}"
    else:
        exp_config['name'] = f"seed{random_seed}"
    
    manager = ExperimentManager(config, **exp_config)
    print(f"\nExperiment: {manager.run_id}")
    print(f"Output dir: {manager.run_dir}")

    # Setup headless mode if specified in config
    eval_cfg = config.get('evaluator', {})
    headless = eval_cfg.get('headless', True)

    # Create configurations from YAML
    memory_cfg = create_memory_config(config)
    eval_config = create_evaluator_config(config)
    
    # Setup pygame for headless mode if needed
    if headless or eval_config.render_mode == "rgb_array":
        import os
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        print("Running in headless mode (no windows will be displayed)")
    else:
        print("Running with human rendering (windows will be displayed)")
    
    pop_config = create_population_config(config)
    evolution_config = create_evolution_config(config, manager)  # Pass manager for paths
    ops_config = get_operations_config(config)

    print(f"\nMemory config: {memory_cfg}")
    
    # Setup operations based on config
    if ops_config.get('use_feature_vector_ops', False):
        all_ops = FEATURE_VECTOR_OPS
        print(f"\nUsing FEATURE_VECTOR operation set: {len(all_ops)} operations")
        print("  - 8 scalar ops: add, sub, mul, div, cos, log, exp, conditional")
        print("  - 4 vector ops: dot_product, mean, sum, norm")
        print("  (Optimized for feature_vector strategy)")
    elif ops_config.get('use_minimal_scalar', False):
        all_ops = MINIMAL_SCALAR_OPS
        print(f"\nUsing MINIMAL SCALAR operation set: {len(all_ops)} operations")
        print("  - add, sub, mul, div (protected)")
        print("  - cos, log, exp")
        print("  - conditional (if-then-else)")
        print("  (Scalar-only for feature_vector strategy)")
    elif ops_config.get('use_minimal', False):
        all_ops = FLAPPYBIRD_MINIMAL_OPS
        print(f"\nUsing MINIMAL operation set: {len(all_ops)} operations")
        print("  - 9 scalar arithmetic ops (add, sub, mul, div, min, max, abs, heaviside, conditional)")
        print("  - 3 scalar trig ops (sin, cos, arctan)")
        print("  - 5 vector ops (intermediate processing)")
        print("  - 6 matrix ops (image manipulation)")
        print("  - 5 CV ops (feature extraction)")
    else:
        all_ops = []
        if ops_config.get('use_automl_no_random', False):
            all_ops.extend(AUTOML_NO_RANDOM_OPS)
            print(f"\nUsing AutoML operations (without random/gaussian/constant): {len(AUTOML_NO_RANDOM_OPS)} operations")
        elif ops_config['use_automl']:
            all_ops.extend(AUTOML_ALL_OPS)
        if ops_config['use_cv']:
            all_ops.extend(CV_ALL_OPS)
        print(f"\nTotal operations: {len(all_ops)}")
        if ops_config.get('use_automl_no_random', False):
            print(f"  - AutoML operations (no random): {len(AUTOML_NO_RANDOM_OPS)}")
        elif ops_config['use_automl']:
            print(f"  - AutoML operations: {len(AUTOML_ALL_OPS)}")
        if ops_config['use_cv']:
            print(f"  - CV operations: {len(CV_ALL_OPS)}")
    
    instruction_set = InstructionSet([op() for op in all_ops], memory_cfg)
    operators = GeneticOperators(instruction_set, rng)

    # Create FlappyBird evaluator
    evaluator = FlappyBirdEvaluator(config=eval_config)

    print("\nFlappyBird Evaluator created!")
    print(f"  Strategy: {evaluator.patch_strategy}")
    print(f"  Color channel: {evaluator.color_channel} (G)")
    print(f"  Episodes per evaluation: {evaluator.episodes}")

    # Create population
    population = Population(
        pop_config,
        instruction_set,
        memory_cfg,
        operators=operators,
        rng=rng,
    )
    population.initialize_random(mutate_constants=True)

    print("\nEvolution Setup Complete!")
    print(f"Population size: {pop_config.size}")
    print(f"Program length range: {pop_config.program_length}")
    print(f"Episodes per evaluation: {evaluator.episodes}")
    print(f"Max steps per episode: {evaluator.max_steps}")
    if evaluator.config and evaluator.config.n_jobs is not None:
        if evaluator.config.n_jobs == 1:
            print(f"Parallelization: Sequential (n_jobs=1)")
        else:
            print(f"Parallelization: {evaluator.config.n_jobs} workers")
    else:
        from multiprocessing import cpu_count
        print(f"Parallelization: Auto-detect ({cpu_count()} CPUs available)")

    # Create and run evolution engine
    engine = EvolutionEngine(
        population=population,
        operators=operators,
        evaluator=evaluator,
        config=evolution_config,
        rng=rng,
    )

    print("\nStarting evolution...")
    print("=" * 60)
    
    # Time the evolution cycle
    start_time = time.time()
    modes_tracker = MODESTracker(
        filter_length=pop_config.elitism,  # paper recommends population size
        output_registers=getattr(evaluator, "output_registers", None),
    )
    final_population = engine.run(modes_tracker=modes_tracker)

    # After engine.run(), save the results:
    try:
        modes_df = modes_tracker.to_dataframe()
        modes_csv_path = manager.run_dir / "modes_metrics.csv"
        modes_df.to_csv(modes_csv_path, index=False)
        print(f"MODES metrics saved to: {modes_csv_path}")
    except ImportError:
        print("pandas not installed — MODES history not saved as CSV")
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("=" * 60)
    print("\nEvolution complete!")
    print(f"Total evolution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    if evolution_config.max_generations > 0:
        avg_time_per_gen = elapsed_time / evolution_config.max_generations
        print(f"Average time per generation: {avg_time_per_gen:.2f} seconds")

    # Close evaluator
    evaluator.close()

    # Get and display best agent
    best_agent = get_best_agent(final_population)
    if best_agent:
        print_best_agent_info(best_agent, final_population, evaluator, memory_cfg)
        
        # Save best agent via manager
        best_gen = final_population.best_ever_generation or final_population.generation
        agent_path = manager.save_best_agent(best_agent, best_gen)
        print(f"\nBest agent saved to: {agent_path}")

    # Generate and save fitness chart
    print("\nGenerating fitness chart...")
    fig = create_fitness_chart(engine)
    if fig is not None:
        chart_path = manager.save_chart(fig, "fitness_evolution")
        plt.close(fig)
        print(f"Fitness chart saved to: {chart_path}")
        print_evolution_summary(engine)

    # Finalize experiment
    best_fitness = best_agent.fitness if best_agent and best_agent.fitness else 0.0
    best_gen = final_population.best_ever_generation or 0
    final_gen = final_population.generation
    manager.finalize(best_fitness, best_gen, final_gen)
    
    print(f"\nExperiment complete!")
    print(f"Results saved to: {manager.run_dir}")
    
    return best_agent


def get_best_agent(population):
    """Get the best agent from the population.
    
    Returns:
        Individual: The best agent (best_ever if available, otherwise best from current generation)
    """
    if population.best_ever is not None:
        return population.best_ever
    else:
        return population.get_best()


def print_best_agent_info(best_agent, population, evaluator, memory_cfg):
    """Print detailed information about the best agent."""
    print(f"\n{'='*70}")
    print("BEST AGENT INFORMATION")
    print(f"{'='*70}")
    print(f"Fitness: {best_agent.fitness:.4f}")
    print(f"Program length: {len(best_agent.program)}")
    print(f"Age: {best_agent.age}")
    
    if population.best_ever is not None and best_agent.id == population.best_ever.id:
        print(f"Best ever generation: {population.best_ever_generation}")
    
    # Get output registers for effective length
    output_registers = getattr(evaluator, "output_registers", None)
    if output_registers:
        effective_length = best_agent.get_effective_length(output_registers)
        effective_ratio = effective_length / len(best_agent.program) if len(best_agent.program) > 0 else 0.0
        print(f"Effective length: {effective_length}")
        print(f"Effective code rate: {effective_ratio:.3f}")
    
    print(f"\nProgram instructions:")
    for i, instr in enumerate(best_agent.program.instructions):
        print(f"  {i:3d}: {instr.to_resolved_str(memory_cfg)}")
    print(f"{'='*70}")


def create_fitness_chart(engine):
    """Create fitness chart from evolution results.
    
    Args:
        engine: EvolutionEngine object containing evolution history
        
    Returns:
        matplotlib Figure object, or None if no data
    """
    if len(engine.best_agent_history) == 0:
        print("No data to plot.")
        return None

    generations = [info.generation for info in engine.best_agent_history]
    fitnesses = [info.fitness for info in engine.best_agent_history]
    code_rates = [info.effective_code_rate for info in engine.best_agent_history]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Best fitness over generations
    ax1 = axes[0, 0]
    ax1.plot(generations, fitnesses, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Best Fitness', fontsize=12)
    ax1.set_title('Best Agent Fitness Over Generations', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.5, max(generations) + 0.5)

    # Plot 2: Effective code rate
    ax2 = axes[0, 1]
    ax2.plot(generations, code_rates, 'r-o', linewidth=2, markersize=6)
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Effective Code Rate', fontsize=12)
    ax2.set_title('Best Agent Effective Code Rate', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)
    ax2.set_xlim(-0.5, max(generations) + 0.5)

    # Plot 3: Population fitness statistics (if available)
    if hasattr(engine.population, 'fitness_history') and len(engine.population.fitness_history) > 0:
        ax3 = axes[1, 0]
        gen_range = range(len(engine.population.fitness_history))
        min_fits = [stats[0] for stats in engine.population.fitness_history]
        mean_fits = [stats[1] for stats in engine.population.fitness_history]
        max_fits = [stats[2] for stats in engine.population.fitness_history]
        
        ax3.plot(gen_range, min_fits, 'g-', label='Min', linewidth=1.5, alpha=0.7)
        ax3.plot(gen_range, mean_fits, 'b-', label='Mean', linewidth=2)
        ax3.plot(gen_range, max_fits, 'r-', label='Max', linewidth=1.5, alpha=0.7)
        ax3.fill_between(gen_range, min_fits, max_fits, alpha=0.2)
        ax3.set_xlabel('Generation', fontsize=12)
        ax3.set_ylabel('Fitness', fontsize=12)
        ax3.set_title('Population Fitness Statistics', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Plot 4: Best agent program length
    ax4 = axes[1, 1]
    total_lengths = [info.total_length for info in engine.best_agent_history]
    effective_lengths = [info.effective_length for info in engine.best_agent_history]
    
    ax4.plot(generations, total_lengths, 'b-o', label='Total Length', linewidth=2, markersize=6)
    ax4.plot(generations, effective_lengths, 'r-o', label='Effective Length', linewidth=2, markersize=6)
    ax4.set_xlabel('Generation', fontsize=12)
    ax4.set_ylabel('Program Length', fontsize=12)
    ax4.set_title('Best Agent Program Length', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-0.5, max(generations) + 0.5)

    plt.tight_layout()
    return fig


def print_evolution_summary(engine):
    """Print evolution summary statistics."""
    generations = [info.generation for info in engine.best_agent_history]
    fitnesses = [info.fitness for info in engine.best_agent_history]
    code_rates = [info.effective_code_rate for info in engine.best_agent_history]
    
    print(f"\n{'='*60}")
    print("EVOLUTION SUMMARY")
    print(f"{'='*60}")
    print(f"Best overall fitness: {max(fitnesses):.4f} (Generation {generations[fitnesses.index(max(fitnesses))]})")
    print(f"Final fitness: {fitnesses[-1]:.4f}")
    print(f"Final effective code rate: {code_rates[-1]:.3f}")
    print(f"Total generations: {len(generations)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run FlappyBird evolution')
    parser.add_argument('config', nargs='?', default='config.yaml',
                       help='Path to YAML configuration file (default: config.yaml)')
    parser.add_argument('--seed', type=int, required=True,
                       help='Random seed (required)')
    args = parser.parse_args()
    main(config_path=args.config, random_seed=args.seed)
