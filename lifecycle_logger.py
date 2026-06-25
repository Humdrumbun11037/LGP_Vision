"""Lifecycle Logger — tracks adaptive mutation rates at individual birth and death.

Each row in the CSV corresponds to one individual's complete life:
  - id, parent_ids, generation_born, generation_died
  - birth_rate_* : sigmoid(register value) at the moment of creation
  - death_rate_* : sigmoid(register value) at the moment of replacement
  - fitness       : evaluated fitness (None if never evaluated)
  - program_length: total instruction count

Usage (in evolution_engine.py or run script)::

    logger = LifecycleLogger(
        path="experiments/my_run/lifecycle.csv",
        adaptive_rate_names=ADAPTIVE_RATE_NAMES,
    )

    # Call when a new individual is created (before evaluation):
    logger.record_birth(individual, generation)

    # Call when an individual is removed from the population:
    logger.record_death(individual, generation)

    logger.close()
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from scipy.special import expit  # sigmoid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_rates(individual, base_index: int, n_registers: int) -> List[float]:
    """Read `n_registers` scalar values starting at `base_index` and sigmoid-map them."""
    return [
        float(expit(individual.memory.read_scalar(base_index + offset)))
        for offset in range(n_registers)
    ]


# ---------------------------------------------------------------------------
# LifecycleLogger
# ---------------------------------------------------------------------------

class LifecycleLogger:
    """Logs per-individual adaptive mutation rates at birth and at death.

    Parameters
    ----------
    path : str or Path
        Destination CSV file.  Parent directories are created automatically.
    adaptive_rate_names : list of str
        Human-readable names for each rate register, e.g.
        ``["micro_mutation", "add_instruction", "delete_instruction", "swap_mutation"]``.
    adaptive_rate_base_index : int
        Index of the first adaptive rate scalar register (default 1, i.e.
        register 0 is the output register).
    buffer_size : int
        Number of rows to accumulate in memory before flushing to disk.
        Larger values reduce I/O overhead; 0 means write every row.
    """

    def __init__(
        self,
        path: Union[str, Path],
        adaptive_rate_names: List[str],
        adaptive_rate_base_index: int = 1,
        buffer_size: int = 200,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rate_names = adaptive_rate_names
        self._base_index = adaptive_rate_base_index
        self._n_rates = len(adaptive_rate_names)
        self._buffer_size = buffer_size

        # Pending rows keyed by individual id; written to disk on death
        self._pending: dict[int, dict] = {}

        # In-memory write buffer (rows ready for disk)
        self._buffer: List[dict] = []

        # Open file and write header
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames())
        self._writer.writeheader()
        self._file.flush()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_birth(self, individual, generation: int) -> None:
        """Snapshot the individual's adaptive rates at the moment of birth.

        Call this immediately after an individual is created (before evaluation).

        Parameters
        ----------
        individual : Individual
        generation : int
            Current generation number.
        """
        rates = _read_rates(individual, self._base_index, self._n_rates)
        self._pending[individual.id] = {
            "id": individual.id,
            "parent_ids": str(individual.parent_ids),
            "generation_born": generation,
            "generation_died": None,
            "fitness": None,
            "program_length": len(individual.program),
            **{f"birth_rate_{name}": rates[i] for i, name in enumerate(self._rate_names)},
            **{f"death_rate_{name}": None for name in self._rate_names},
        }

    def record_death(self, individual, generation: int) -> None:
        """Snapshot the individual's adaptive rates at the moment of death.

        Call this for each individual that is removed from the population
        (i.e. not carried forward as an elite or offspring).

        Parameters
        ----------
        individual : Individual
        generation : int
            Generation at which the individual is replaced / removed.
        """
        rates = _read_rates(individual, self._base_index, self._n_rates)
        row = self._pending.pop(individual.id, None)

        if row is None:
            # Individual was never birth-logged (e.g. from the initial population)
            # — create a synthetic record with unknown birth info.
            row = {
                "id": individual.id,
                "parent_ids": str(individual.parent_ids),
                "generation_born": "?",
                "generation_died": generation,
                "fitness": individual.fitness,
                "program_length": len(individual.program),
                **{f"birth_rate_{name}": None for name in self._rate_names},
                **{f"death_rate_{name}": rates[i] for i, name in enumerate(self._rate_names)},
            }
        else:
            row["generation_died"] = generation
            row["fitness"] = individual.fitness
            row["program_length"] = len(individual.program)
            for i, name in enumerate(self._rate_names):
                row[f"death_rate_{name}"] = rates[i]

        self._buffer.append(row)
        if self._buffer_size == 0 or len(self._buffer) >= self._buffer_size:
            self._flush()

    def close(self) -> None:
        """Flush remaining buffer and close the file."""
        # Flush any individuals that were born but never explicitly died
        # (e.g. surviving elites at the end of the run)
        for ind_id, row in self._pending.items():
            row["generation_died"] = "survived"
            self._buffer.append(row)
        self._pending.clear()
        self._flush()
        self._file.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fieldnames(self) -> List[str]:
        return (
            ["id", "parent_ids", "generation_born", "generation_died",
             "fitness", "program_length"]
            + [f"birth_rate_{n}" for n in self._rate_names]
            + [f"death_rate_{n}" for n in self._rate_names]
        )

    def _flush(self) -> None:
        if not self._buffer:
            return
        self._writer.writerows(self._buffer)
        self._file.flush()
        self._buffer.clear()