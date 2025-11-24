#!/usr/bin/env python3
"""Run FlappyBird evolution and generate fitness chart."""

import flappy_bird_env  # noqa
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

import gymnasium as gym
from memory_system import MemoryConfig
from instruction_set import InstructionSet
from operation import AUTOML_ALL_OPS, CV_ALL_OPS
from population import Population, PopulationConfig
from operators import GeneticOperators
from evaluator import FlappyBirdEvaluator
from evolution_engine import EvolutionEngine, EvolutionConfig


def main():
    """Run FlappyBird evolution."""
    # Setup random seed
    rng = np.random.default_rng(0)

    # Memory configuration for FlappyBird
    patch_size = 37  # Based on quantization_factor=0.05
    memory_cfg = MemoryConfig(
        n_scalar=8,          # Working scalars (for computation)
        n_vector=8,          # Working vectors
        n_matrix=8,          # Working matrices (for CV operations)
        n_obs_scalar=0,      # No scalar observations (using matrices instead)
        n_obs_vector=0,      # No vector observations
        n_obs_matrix=1,      # Matrix observation register
        vector_size=37,
        matrix_shape=(patch_size, patch_size),
    )

    # Use AutoML + CV operations for image processing
    all_ops = AUTOML_ALL_OPS + CV_ALL_OPS
    instruction_set = InstructionSet([op() for op in all_ops], memory_cfg)
    operators = GeneticOperators(instruction_set, rng)

    print(f"Memory config: {memory_cfg}")
    print(f"Total operations: {len(all_ops)}")
    print(f"  - AutoML operations: {len(AUTOML_ALL_OPS)}")
    print(f"  - CV operations: {len(CV_ALL_OPS)}")

    # Create FlappyBird evaluator for evolution (no rendering for speed)
    evaluator = FlappyBirdEvaluator(
        env_id="FlappyBird-v0",
        episodes=5,  # Number of episodes per evaluation
        max_steps=500,
        output_register=0,  # Read action from scalar register 0
        render_mode="rgb_array",   # No rendering for speed
        rng=rng,
        # Image processing parameters
        patch_strategy="quantized",
        color_channel=1,  # Green channel
        normalize=True,
        quantization_factor=0.05,
    )

    print("\nFlappyBird Evaluator created!")
    print(f"  Strategy: {evaluator.patch_strategy}")
    print(f"  Color channel: {evaluator.color_channel} (G)")
    print(f"  Episodes per evaluation: {evaluator.episodes}")

    # Population configuration
    pop_config = PopulationConfig(
        size=5,
        program_length=(1, 10),
        elitism=1,
        max_program_length=250,
    )

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

    # Evolution configuration
    evolution_config = EvolutionConfig(
        max_generations=5,
        mutation_threshold=0.9,
        constant_mutation_rate=0.1,
        verbose=True,
        checkpoint_path="checkpoints/best_population.pkl",
        stats_log_dir="stats_log",
    )

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
    final_population = engine.run()
    print("=" * 60)
    print("\nEvolution complete!")

    # Close evaluator
    evaluator.close()

    # Get and display best agent
    best_agent = get_best_agent(final_population)
    if best_agent:
        print_best_agent_info(best_agent, final_population, evaluator)
        
        # Optionally save best agent to file
        save_best_agent(best_agent, final_population)

    # Generate fitness chart
    print("\nGenerating fitness chart...")
    plot_fitness_chart(engine)

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


def print_best_agent_info(best_agent, population, evaluator):
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
        print(f"  {i:3d}: {instr}")
    print(f"{'='*70}")


def save_best_agent(best_agent, population):
    """Save the best agent to a pickle file."""
    output_dir = Path("best_agents")
    output_dir.mkdir(exist_ok=True)
    
    # Create filename with fitness and generation info
    fitness_str = f"{best_agent.fitness:.4f}".replace(".", "_")
    gen_str = ""
    if population.best_ever is not None and best_agent.id == population.best_ever.id:
        gen_str = f"_gen{population.best_ever_generation}"
    
    filename = f"best_agent_fitness_{fitness_str}{gen_str}.pkl"
    output_path = output_dir / filename
    
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(best_agent, f)
        print(f"\nBest agent saved to: {output_path}")
    except Exception as e:
        print(f"\nWarning: Failed to save best agent: {e}")


def plot_fitness_chart(engine):
    """Plot fitness chart from evolution results."""
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
    output_path = Path("fitness_chart.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Fitness chart saved to: {output_path}")
    
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
    main()

