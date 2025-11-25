#!/usr/bin/env python3
"""Run FlappyBird evolution and generate fitness chart."""

import flappy_bird_env  # noqa
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

import gymnasium as gym
from memory_system import MemoryConfig
from instruction_set import InstructionSet
from operation import AUTOML_ALL_OPS, CV_ALL_OPS, FLAPPYBIRD_MINIMAL_OPS
from population import Population, PopulationConfig
from operators import GeneticOperators
from evaluator import FlappyBirdEvaluator, FlappyBirdEvaluatorConfig
from evolution_engine import EvolutionEngine, EvolutionConfig

# Import config loader
from config_loader import (
    load_config,
    create_memory_config,
    create_evaluator_config,
    create_population_config,
    create_evolution_config,
    get_operations_config,
    get_output_config,
)


def main(config_path: str = "config.yaml"):
    """Run FlappyBird evolution.
    
    Args:
        config_path: Path to YAML configuration file
    """
    # Load configuration from YAML
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    # Setup random seed from config
    random_seed = config.get('random_seed', 42)
    rng = np.random.default_rng(random_seed)
    print(f"Random seed: {random_seed}")

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
    evolution_config = create_evolution_config(config)
    ops_config = get_operations_config(config)
    output_config = get_output_config(config)

    print(f"\nMemory config: {memory_cfg}")
    
    # Setup operations based on config
    if ops_config.get('use_minimal', False):
        # Use carefully selected minimal 27-operation set
        all_ops = FLAPPYBIRD_MINIMAL_OPS
        print(f"\nUsing MINIMAL operation set: {len(all_ops)} operations")
        print("  - 8 scalar arithmetic ops (add, sub, mul, div, min, max, abs, heaviside)")
        print("  - 3 scalar trig ops (sin, cos, arctan)")
        print("  - 5 vector ops (intermediate processing)")
        print("  - 6 matrix ops (image manipulation)")
        print("  - 5 CV ops (feature extraction)")
    else:
        # Use full operation sets
        all_ops = []
        if ops_config['use_automl']:
            all_ops.extend(AUTOML_ALL_OPS)
        if ops_config['use_cv']:
            all_ops.extend(CV_ALL_OPS)
        print(f"\nTotal operations: {len(all_ops)}")
        if ops_config['use_automl']:
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
    final_population = engine.run()
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
        
        # Save best agent if configured
        if output_config.get('save_best_agent', True):
            save_best_agent(
                best_agent, 
                final_population, 
                output_config.get('best_agent_dir', 'best_agents')
            )

    # Generate fitness chart
    print("\nGenerating fitness chart...")
    plot_fitness_chart(engine, output_config.get('fitness_chart_path', 'fitness_chart.png'))

    print("\nRun complete!")
    
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


def print_best_agent_info(best_agent, population, evaluator,memory_cfg):
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


def save_best_agent(best_agent, population, output_dir="best_agents"):
    """Save the best agent to a pickle file.
    
    Args:
        best_agent: The best agent Individual to save
        population: Population object containing generation info
        output_dir: Directory to save the agent file (default: "best_agents")
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Create filename with fitness and generation info
    fitness_str = f"{best_agent.fitness:.4f}".replace(".", "_")
    gen_str = ""
    if population.best_ever is not None and best_agent.id == population.best_ever.id:
        gen_str = f"_gen{population.best_ever_generation}"
    
    filename = f"best_agent_fitness_{fitness_str}{gen_str}.pkl"
    file_path = output_path / filename
    
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(best_agent, f)
        print(f"\nBest agent saved to: {file_path}")
    except Exception as e:
        print(f"\nWarning: Failed to save best agent: {e}")


def plot_fitness_chart(engine, output_path="fitness_chart.png"):
    """Plot fitness chart from evolution results.
    
    Args:
        engine: EvolutionEngine object containing evolution history
        output_path: Path to save the fitness chart (default: "fitness_chart.png")
    """
    if len(engine.best_agent_history) == 0:
        print("No data to plot.")
        return

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
    
    # Save figure
    chart_path = Path(output_path)
    chart_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"Fitness chart saved to: {chart_path}")
    
    # Also display if possible
    try:
        plt.show()
    except Exception:
        print("(Display not available, chart saved to file)")

    # Print summary statistics
    print(f"\n{'='*60}")
    print("EVOLUTION SUMMARY")
    print(f"{'='*60}")
    print(f"Best overall fitness: {max(fitnesses):.4f} (Generation {generations[fitnesses.index(max(fitnesses))]})")
    print(f"Final fitness: {fitnesses[-1]:.4f}")
    print(f"Final effective code rate: {code_rates[-1]:.3f}")
    print(f"Total generations: {len(generations)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Allow config file to be passed as command-line argument
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(config_path=config_file)

