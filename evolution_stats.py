"""Evolution Statistics Display

A clean, reusable class for displaying evolution statistics and visualizations.
Just run: stats_display = EvolutionStatsDisplay(engine); stats_display.display()
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from evolution_engine import EvolutionEngine
    from population import Population
    from memory_system import MemoryConfig


class EvolutionStatsDisplay:
    """Displays comprehensive evolution statistics and visualizations."""
    
    def __init__(
        self,
        engine: Optional['EvolutionEngine'] = None,
        population: Optional['Population'] = None,
        best_agent_history: Optional[List] = None,
        memory_cfg: Optional['MemoryConfig'] = None,
        output_registers: Optional[List[Tuple]] = None
    ):
        """
        Initialize with either:
        - engine: EvolutionEngine (preferred, extracts everything automatically)
        - OR manually: population, best_agent_history, memory_cfg, output_registers
        """
        if engine is not None:
            self.population = engine.population
            self.best_agent_history = engine.best_agent_history
            self.memory_cfg = engine.population.memory_config
            self.output_registers = getattr(engine.evaluator, 'output_registers', None)
            self.adaptive_rate_history = engine.adaptive_rate_history
            self.adaptive_mutation_rates = engine.config.adaptive_mutation_rates
        else:
            self.population = population
            self.best_agent_history = best_agent_history or []
            self.memory_cfg = memory_cfg
            self.output_registers = output_registers
            self.adaptive_rate_history = []
            self.adaptive_mutation_rates = False
        
        if self.population is not None:
            self.best_agent = self.population.best_ever or self.population.get_best()
        else:
            self.best_agent = None
    
    def display(self, show_code: bool = True, show_graphs: bool = True, show_stats: bool = True):
        """Display all statistics and visualizations.
        
        Args:
            show_code: Whether to display best agent code
            show_graphs: Whether to display fitness and program length graphs
            show_stats: Whether to display summary statistics
        """
        if show_stats:
            self._print_summary_stats()
        
        if show_graphs:
            self._plot_fitness_evolution()
            self._plot_program_length_evolution()
            if self.adaptive_mutation_rates and self.adaptive_rate_history:
                self._plot_adaptive_rates()
        
        if show_code and self.best_agent is not None:
            self._print_best_agent_code()
    
    def _print_summary_stats(self):
        """Print summary statistics."""
        print("=" * 80)
        print("EVOLUTION SUMMARY STATISTICS")
        print("=" * 80)
        
        if self.population is None:
            print("⚠️  No population data available")
            return
        
        min_fit, mean_fit, max_fit, std_fit = self.population.compute_statistics()
        
        print(f"\n📊 Current Generation ({self.population.generation}):")
        print(f"   Population size: {len(self.population.individuals)}")
        print(f"   Min fitness:   {min_fit:.4f}")
        print(f"   Mean fitness:  {mean_fit:.4f}")
        print(f"   Max fitness:   {max_fit:.4f}")
        print(f"   Std fitness:   {std_fit:.4f}")
        
        if self.best_agent is not None and self.best_agent.fitness is not None:
            print(f"\n🏆 Best Agent:")
            print(f"   Fitness:      {self.best_agent.fitness:.4f}")
            print(f"   Program length: {len(self.best_agent.program)}")
            print(f"   Age:          {self.best_agent.age}")
            print(f"   ID:           {self.best_agent.id}")
            if self.best_agent.parent_ids:
                print(f"   Parent IDs:   {self.best_agent.parent_ids}")
            
            if self.population.best_ever is not None:
                print(f"\n⭐ Best Ever:")
                print(f"   Fitness:      {self.population.best_ever.fitness:.4f}")
                print(f"   Generation:   {self.population.best_ever_generation}")

            # Print current adaptive rates of the best agent
            if self.adaptive_mutation_rates and self.adaptive_rate_history:
                from evolution_engine import ADAPTIVE_RATE_NAMES
                stats = self.adaptive_rate_history[-1]
                print(f"\n🧬 Best Agent Adaptive Mutation Rates (sigmoid-mapped):")
                for i, name in enumerate(ADAPTIVE_RATE_NAMES):
                    print(f"   {name}: {stats.best[i]:.4f}  "
                          f"(pop mean={stats.mean[i]:.4f} ± {stats.std[i]:.4f})")
        
        if len(self.best_agent_history) > 1:
            initial_fitness = self.best_agent_history[0].fitness
            final_fitness = self.best_agent_history[-1].fitness
            improvement = final_fitness - initial_fitness
            improvement_pct = (improvement / abs(initial_fitness) * 100) if initial_fitness != 0 else 0
            
            print(f"\n📈 Evolution Progress:")
            print(f"   Generations:  {len(self.best_agent_history)}")
            print(f"   Initial fitness: {initial_fitness:.4f}")
            print(f"   Final fitness:   {final_fitness:.4f}")
            print(f"   Improvement:     {improvement:+.4f} ({improvement_pct:+.2f}%)")
        
        print("=" * 80)
    
    def _plot_fitness_evolution(self):
        """Plot fitness evolution across generations."""
        if len(self.best_agent_history) == 0:
            print("⚠️  No evolution history available for plotting")
            return
        
        generations = [info.generation for info in self.best_agent_history]
        best_fitnesses = [info.fitness for info in self.best_agent_history]
        
        min_fits = []
        mean_fits = []
        max_fits = []
        
        if self.population is not None and hasattr(self.population, 'fitness_history') and len(self.population.fitness_history) > 0:
            min_fits = [stats[0] for stats in self.population.fitness_history]
            mean_fits = [stats[1] for stats in self.population.fitness_history]
            max_fits = [stats[2] for stats in self.population.fitness_history]
            gen_range = list(range(len(self.population.fitness_history)))
        else:
            min_fits = [min(best_fitnesses)] * len(generations)
            mean_fits = best_fitnesses
            max_fits = best_fitnesses
            gen_range = generations
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Fitness Evolution Across Generations', fontsize=14, fontweight='bold')
        
        ax1 = axes[0]
        ax1.plot(gen_range, min_fits, 'g-', label='Min Fitness', linewidth=1.5, alpha=0.6, marker='o', markersize=3)
        ax1.plot(gen_range, mean_fits, 'b-', label='Mean Fitness', linewidth=2, marker='s', markersize=4)
        ax1.plot(gen_range, max_fits, 'orange', label='Max Fitness', linewidth=1.5, alpha=0.6, marker='^', markersize=3)
        ax1.plot(generations, best_fitnesses, 'purple', linewidth=3, marker='o', markersize=6, 
                label='Best Agent Fitness', zorder=4)
        ax1.fill_between(gen_range, min_fits, max_fits, alpha=0.15, color='gray', label='Population Range')
        
        if self.population is not None and self.population.best_ever is not None:
            best_ever_gen = self.population.best_ever_generation
            best_ever_fit = self.population.best_ever.fitness
            ax1.plot(best_ever_gen, best_ever_fit, 'r*', markersize=20, 
                    label=f'Best Ever ({best_ever_fit:.2f})', zorder=5, 
                    markeredgecolor='darkred', markeredgewidth=2)
        
        ax1.set_xlabel('Generation', fontsize=11)
        ax1.set_ylabel('Fitness', fontsize=11)
        ax1.set_title('Fitness Statistics', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        ax2.plot(generations, best_fitnesses, 'purple', linewidth=3, marker='o', markersize=6, 
                label='Best Agent Fitness')
        if self.population is not None and self.population.best_ever is not None:
            best_ever_gen = self.population.best_ever_generation
            best_ever_fit = self.population.best_ever.fitness
            ax2.plot(best_ever_gen, best_ever_fit, 'r*', markersize=20, 
                    label=f'Best Ever ({best_ever_fit:.2f})', zorder=5,
                    markeredgecolor='darkred', markeredgewidth=2)
        
        ax2.set_xlabel('Generation', fontsize=11)
        ax2.set_ylabel('Fitness', fontsize=11)
        ax2.set_title('Best Agent Fitness Over Time', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_program_length_evolution(self):
        """Plot program length evolution."""
        if len(self.best_agent_history) == 0:
            return
        
        generations = [info.generation for info in self.best_agent_history]
        total_lengths = [info.total_length for info in self.best_agent_history]
        effective_lengths = [info.effective_length for info in self.best_agent_history]
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        ax.plot(generations, total_lengths, 'b-', label='Total Length', linewidth=2, marker='o', markersize=5)
        ax.plot(generations, effective_lengths, 'g-', label='Effective Length', linewidth=2, marker='s', markersize=5)
        
        ax.set_xlabel('Generation', fontsize=11)
        ax.set_ylabel('Program Length', fontsize=11)
        ax.set_title('Program Length Evolution', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def _plot_adaptive_rates(self):
        """
        Plot population statistics and best-agent values for each adaptive
        mutation rate register across generations.

        Layout: 2 rows × 2 columns, one subplot per rate register.
        """
        if not self.adaptive_rate_history:
            return

        from evolution_engine import ADAPTIVE_RATE_NAMES, N_ADAPTIVE_RATE_REGISTERS

        history = self.adaptive_rate_history
        generations = [s.generation for s in history]
        colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Adaptive Mutation Rates Across Generations',
                     fontsize=15, fontweight='bold')

        for i, (ax, name, colour) in enumerate(
            zip(axes.flatten(), ADAPTIVE_RATE_NAMES, colours)
        ):
            mean_vals = np.array([s.mean[i] for s in history])
            std_vals  = np.array([s.std[i]  for s in history])
            min_vals  = [s.min[i]  for s in history]
            max_vals  = [s.max[i]  for s in history]
            best_vals = [s.best[i] for s in history]

            ax.fill_between(generations,
                            np.clip(mean_vals - std_vals, 0, 1),
                            np.clip(mean_vals + std_vals, 0, 1),
                            alpha=0.30, color=colour, label='Pop mean ± std')
            ax.plot(generations, mean_vals, color=colour, linewidth=2, label='Pop mean')
            ax.plot(generations, best_vals, color=colour, linewidth=2, linestyle='--',
                    marker='o', markersize=3, label='Best agent')

            ax.set_xlim(generations[0], generations[-1])
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel('Generation', fontsize=11)
            ax.set_ylabel('Rate (sigmoid)', fontsize=11)
            ax.set_title(f'Register {i + 1}: {name.replace("_", " ").title()}',
                         fontsize=12, fontweight='bold')
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def _print_best_agent_code(self):
        """Print the best agent's code."""
        if self.best_agent is None:
            print("⚠️  No best agent available")
            return
        
        print("\n" + "=" * 80)
        print("BEST AGENT CODE")
        print("=" * 80)
        
        print(f"\nAgent ID: {self.best_agent.id}")
        print(f"Fitness: {self.best_agent.fitness:.4f}")
        print(f"Program Length: {len(self.best_agent.program)}")
        print(f"Age: {self.best_agent.age}")
        
        print(f"\n{'='*80}")
        print("FULL PROGRAM")
        print("=" * 80)
        
        if self.memory_cfg is not None:
            for i, instr in enumerate(self.best_agent.program.instructions):
                print(f"  {i:3d}: {instr.to_resolved_str(self.memory_cfg)}")
        else:
            for i, instr in enumerate(self.best_agent.program.instructions):
                print(f"  {i:3d}: {instr}")
        
        print(f"\n{'='*80}")
        print("EVOLVED CONSTANTS (Working Scalars)")
        print("=" * 80)

        if self.adaptive_mutation_rates:
            from evolution_engine import (
                ADAPTIVE_RATE_BASE_INDEX, N_ADAPTIVE_RATE_REGISTERS,
                ADAPTIVE_RATE_NAMES, _read_adaptive_rates,
            )
            rates = _read_adaptive_rates(self.best_agent)

        if self.best_agent.memory.n_scalar > 0:
            for i in range(min(self.best_agent.memory.n_scalar, 20)):
                val = self.best_agent.memory.read_scalar(i)
                markers = []
                if self.output_registers and any(reg[1] == i for reg in self.output_registers):
                    markers.append("← OUTPUT")
                if self.adaptive_mutation_rates:
                    offset = i - ADAPTIVE_RATE_BASE_INDEX
                    if 0 <= offset < N_ADAPTIVE_RATE_REGISTERS:
                        sigmoid_val = rates[offset]
                        markers.append(
                            f"← ADAPTIVE {ADAPTIVE_RATE_NAMES[offset]} "
                            f"(raw={val:.4f}, σ={sigmoid_val:.4f})"
                        )
                marker_str = "  " + "  ".join(markers) if markers else ""
                print(f"  scalar[{i:2d}] = {val:10.6f}{marker_str}")
        
        print("=" * 80)