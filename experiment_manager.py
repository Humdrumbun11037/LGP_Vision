"""Experiment Manager for unified run output organization.

Centralizes all experiment outputs (checkpoints, stats, agents, charts) under
a single timestamped directory with config snapshot and metadata.
"""

from __future__ import annotations

import json
import pickle
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from individual import Individual
    from population import Population
    import matplotlib.figure


class ExperimentManager:
    """Manages experiment outputs with unified naming and organization.
    
    Creates directory structure:
        experiments/
        └── run_20251130_143052_name/
            ├── config.yaml          # Snapshot of config used
            ├── run_info.json        # Metadata (status, results, timing)
            ├── checkpoints/
            │   ├── gen_0000.pkl
            │   ├── gen_0050.pkl
            │   └── latest.pkl       # Symlink to most recent
            ├── stats/
            │   └── generation_stats.csv
            ├── agents/
            │   └── best_agent_gen42_fit0.1234.pkl
            └── charts/
                └── fitness_evolution.png
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        name: Optional[str] = None,
        base_dir: str = "experiments",
        checkpoint_every: Optional[int] = 50,
        keep_n_checkpoints: Optional[int] = None,
    ):
        """Initialize experiment manager.
        
        Args:
            config: Full configuration dictionary (will be saved as snapshot)
            name: Optional custom name suffix for run ID
            base_dir: Base directory for all experiment runs
            checkpoint_every: Save checkpoint every N generations (None = only on improvement)
            keep_n_checkpoints: Keep last N checkpoints (None = keep all)
        """
        self.config = config
        self.base_dir = Path(base_dir)
        self.checkpoint_every = checkpoint_every
        self.keep_n_checkpoints = keep_n_checkpoints
        
        # Generate run ID with timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_suffix = f"_{name}" if name else ""
        self.run_id = f"run_{self.timestamp}{name_suffix}"
        
        # Setup directory structure
        self.run_dir = self.base_dir / self.run_id
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.stats_dir = self.run_dir / "stats"
        self.agents_dir = self.run_dir / "agents"
        self.charts_dir = self.run_dir / "charts"
        
        # Track state
        self._start_time = datetime.now()
        self._checkpoint_files: list[Path] = []
        
        # Initialize
        self._create_directories()
        self._save_config_copy()
        self._init_run_info()
    
    def _create_directories(self) -> None:
        """Create all output directories."""
        for directory in [
            self.checkpoints_dir,
            self.stats_dir,
            self.agents_dir,
            self.charts_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _save_config_copy(self) -> None:
        """Save a copy of the configuration used for this run."""
        config_path = self.run_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
    
    def _init_run_info(self) -> None:
        """Initialize run_info.json with metadata."""
        self.run_info: Dict[str, Any] = {
            "run_id": self.run_id,
            "start_time": self._start_time.isoformat(),
            "end_time": None,
            "status": "running",
            "config_summary": self._extract_config_summary(),
            "results": None,
        }
        self._save_run_info()
    
    def _extract_config_summary(self) -> Dict[str, Any]:
        """Extract key configuration values for quick reference."""
        summary = {}
        
        # Store seed
        summary["random_seed"] = self.config.get("random_seed")
        
        # Population config
        pop_cfg = self.config.get("population", {})
        summary["population_size"] = pop_cfg.get("size")
        summary["program_length"] = pop_cfg.get("program_length")
        summary["elitism"] = pop_cfg.get("elitism")
        
        # Evolution config
        evo_cfg = self.config.get("evolution", {})
        summary["max_generations"] = evo_cfg.get("max_generations")
        summary["mutation_threshold"] = evo_cfg.get("mutation_threshold")
        summary["crossover_threshold"] = evo_cfg.get("crossover_threshold")
        
        # Evaluator config
        eval_cfg = self.config.get("evaluator", {})
        summary["patch_strategy"] = eval_cfg.get("patch_strategy")
        summary["episodes"] = eval_cfg.get("episodes")
        summary["max_steps"] = eval_cfg.get("max_steps")
        summary["n_jobs"] = eval_cfg.get("n_jobs")
        
        # Operations config
        ops_cfg = self.config.get("operations", {})
        if ops_cfg.get("use_feature_vector_ops"):
            summary["operation_set"] = "feature_vector_ops"
        elif ops_cfg.get("use_minimal_scalar"):
            summary["operation_set"] = "minimal_scalar"
        elif ops_cfg.get("use_minimal"):
            summary["operation_set"] = "minimal"
        else:
            summary["operation_set"] = "full"
        
        return summary
    
    def _save_run_info(self) -> None:
        """Save run_info.json to disk."""
        run_info_path = self.run_dir / "run_info.json"
        with open(run_info_path, 'w') as f:
            json.dump(self.run_info, f, indent=2)
    
    def get_stats_csv_path(self) -> Path:
        """Get path for generation statistics CSV file.
        
        Returns:
            Path to stats/generation_stats.csv
        """
        return self.stats_dir / "generation_stats.csv"
    
    def save_checkpoint(
        self,
        population: 'Population',
        generation: int,
        fitness: Optional[float] = None,
        is_final: bool = False,
    ) -> Path:
        """Save a population checkpoint.
        
        Args:
            population: Population object to save
            generation: Current generation number
            fitness: Best fitness at this generation (for metadata)
            is_final: Whether this is the final checkpoint
            
        Returns:
            Path to saved checkpoint file
        """
        # Create filename with generation number
        filename = f"gen_{generation:04d}.pkl"
        checkpoint_path = self.checkpoints_dir / filename
        
        # Prepare checkpoint data
        checkpoint_data = {
            "generation": generation,
            "fitness": fitness,
            "population": population,
            "timestamp": datetime.now().isoformat(),
            "run_id": self.run_id,
        }
        
        # Save checkpoint
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        # Track checkpoint file
        self._checkpoint_files.append(checkpoint_path)
        
        # Update "latest" symlink
        latest_path = self.checkpoints_dir / "latest.pkl"
        if latest_path.exists() or latest_path.is_symlink():
            latest_path.unlink()
        
        # Create relative symlink
        try:
            latest_path.symlink_to(filename)
        except OSError:
            # Symlinks may not work on all systems, just copy instead
            import shutil
            shutil.copy2(checkpoint_path, latest_path)
        
        # Prune old checkpoints if configured
        if self.keep_n_checkpoints is not None and not is_final:
            self._prune_checkpoints()
        
        return checkpoint_path
    
    def _prune_checkpoints(self) -> None:
        """Remove old checkpoints if exceeding keep_n_checkpoints."""
        if self.keep_n_checkpoints is None:
            return
        
        # Filter out "latest.pkl" from the list
        checkpoint_files = [
            f for f in self._checkpoint_files 
            if f.name != "latest.pkl"
        ]
        
        # Remove oldest checkpoints if we have too many
        while len(checkpoint_files) > self.keep_n_checkpoints:
            oldest = checkpoint_files.pop(0)
            if oldest.exists():
                oldest.unlink()
            if oldest in self._checkpoint_files:
                self._checkpoint_files.remove(oldest)
    
    def save_best_agent(
        self,
        agent: 'Individual',
        generation: int,
    ) -> Path:
        """Save the best agent to a pickle file.
        
        Args:
            agent: Best agent Individual to save
            generation: Generation when this agent was found
            
        Returns:
            Path to saved agent file
        """
        # Create filename with generation and fitness
        fitness_str = f"{agent.fitness:.4f}".replace(".", "_") if agent.fitness else "unknown"
        filename = f"best_agent_gen{generation:04d}_fit{fitness_str}.pkl"
        agent_path = self.agents_dir / filename
        
        with open(agent_path, 'wb') as f:
            pickle.dump(agent, f)
        
        return agent_path
    
    def save_chart(
        self,
        fig: 'matplotlib.figure.Figure',
        name: str,
        dpi: int = 150,
    ) -> Path:
        """Save a matplotlib figure.
        
        Args:
            fig: Matplotlib figure to save
            name: Name for the chart file (without extension)
            dpi: Resolution for saved image
            
        Returns:
            Path to saved chart file
        """
        chart_path = self.charts_dir / f"{name}.png"
        fig.savefig(chart_path, dpi=dpi, bbox_inches='tight')
        return chart_path
    
    def finalize(
        self,
        best_fitness: float,
        best_generation: int,
        final_generation: int,
        status: str = "completed",
    ) -> None:
        """Mark run as complete and update metadata.
        
        Args:
            best_fitness: Best fitness achieved
            best_generation: Generation when best fitness was found
            final_generation: Final generation number
            status: Final status ("completed", "failed", "interrupted")
        """
        end_time = datetime.now()
        runtime_seconds = (end_time - self._start_time).total_seconds()
        
        self.run_info["end_time"] = end_time.isoformat()
        self.run_info["status"] = status
        self.run_info["results"] = {
            "best_fitness": best_fitness,
            "best_generation": best_generation,
            "final_generation": final_generation,
            "runtime_seconds": runtime_seconds,
            "runtime_formatted": self._format_duration(runtime_seconds),
        }
        
        self._save_run_info()
    
    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in human-readable format."""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    def should_checkpoint(self, generation: int) -> bool:
        """Check if a checkpoint should be saved at this generation.
        
        Args:
            generation: Current generation number
            
        Returns:
            True if checkpoint should be saved
        """
        if self.checkpoint_every is None:
            return False
        return generation % self.checkpoint_every == 0
    
    @staticmethod
    def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
        """Load a checkpoint file.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dictionary with checkpoint data including 'population', 'generation', etc.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def load_run_info(run_dir: str) -> Dict[str, Any]:
        """Load run_info.json from a run directory.
        
        Args:
            run_dir: Path to experiment run directory
            
        Returns:
            Dictionary with run metadata
        """
        run_info_path = Path(run_dir) / "run_info.json"
        if not run_info_path.exists():
            raise FileNotFoundError(f"run_info.json not found in: {run_dir}")
        
        with open(run_info_path, 'r') as f:
            return json.load(f)
    
    def __repr__(self) -> str:
        return f"ExperimentManager(run_id='{self.run_id}', run_dir='{self.run_dir}')"

