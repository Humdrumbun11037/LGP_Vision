
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load all runs
files = [
    "generation_stats_0.csv",
    "generation_stats_789.csv",
    "generation_stats_42.csv",
    "generation_stats_456.csv",
    "generation_stats_123.csv",


]
runs = [pd.read_csv(f) for f in files]

# Find minimum length across runs
min_len = min(len(r) for r in runs)

# Align all runs to the same length
runs_aligned = [r.iloc[:min_len] for r in runs]

# Stack best_fitness values for each generation
best_fitness_matrix = np.array([r['best_fitness'].values for r in runs_aligned])
best_ever_matrix = np.array([r['best_ever_fitness'].values for r in runs_aligned])

# Compute statistics using median and std
median_best_fitness = np.median(best_fitness_matrix, axis=0)
std_best_fitness = np.std(best_fitness_matrix, axis=0)

# Best agent (max across runs for each generation)
best_agent = np.max(best_ever_matrix, axis=0)

generations = np.arange(min_len)

# Create figure
fig, ax = plt.subplots(figsize=(12, 7))

# Plot median best fitness with std band
ax.plot(generations, median_best_fitness, label='Median Best Fitness', color='#2196F3', linewidth=2)
ax.fill_between(generations, 
                median_best_fitness - std_best_fitness, 
                median_best_fitness + std_best_fitness, 
                alpha=0.25, color='#2196F3', label='±1 Std Dev')

# Plot best agent (max across all runs)
ax.plot(generations, best_agent, label='Best Agent', color='#FF5722', linewidth=2, linestyle='--')

# Labels and styling
ax.set_xlabel('Generation', fontsize=12)
ax.set_ylabel('Fitness', fontsize=12)
ax.set_title('LGP Trinary Runs: Fitness Progression (n=5 runs)', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, min_len-1)

plt.tight_layout()
plt.savefig('trinary_runs_fitness.png', dpi=150, bbox_inches='tight')
print("Saved plot with std")
