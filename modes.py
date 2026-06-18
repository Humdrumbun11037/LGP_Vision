# modes.py
"""
MODES Toolbox — Measurements of Open-Ended Dynamics in Evolving Systems
Dolson et al., Artificial Life 25(1), 2019.
 
This module provides:
  - Metric functions  (change, novelty, ecology, complexity)
  - PersistenceFilter  (lineage-based noise filter)
  - MODESTracker       (stateful tracker to plug into EvolutionEngine)
 
Intended usage
--------------
Create a MODESTracker before your evolution loop and call
`tracker.record(population, generation)` at the end of each generation.
Results are accumulated in `tracker.history` and can be inspected or
plotted at any time.
 
    tracker = MODESTracker(
        filter_length=population_size,
        output_registers=evaluator.output_registers,
    )
    for gen in range(max_generations):
        engine.run_one_generation()
        tracker.record(engine.population, gen)
 
    df = tracker.to_dataframe()
"""
 
from __future__ import annotations
 
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple
 
import numpy as np
from scipy.stats import entropy
 
# ---------------------------------------------------------------------------
# Type alias for a "component" in MODES terminology.
# In this codebase a component is the sequence of effective instructions
# belonging to one individual, encoded as a hashable tuple of strings.
# ---------------------------------------------------------------------------
Component = Tuple[str, ...]
 
 
# ===========================================================================
# 1. Fingerprinting — turning an Individual into a hashable Component
# ===========================================================================
 
def _instruction_to_str(instr) -> str:
    """Stable string representation of a single Instruction."""
    op_name = type(instr.operation).__name__
    dest = f"{instr.dest_type.name}[{instr.dest_index}]"
    srcs = ",".join(
        f"{t.name}[{i}]" for t, i in zip(instr.source_types, instr.source_indices)
    )
    return f"{op_name}:{dest}=({srcs})"
 
 
def get_component(individual, output_registers=None) -> Component:
    """
    Return a hashable fingerprint for *individual*.
 
    If *output_registers* is supplied, only effective (non-intron)
    instructions are included — this matches the paper's recommendation
    to reduce genomes to their meaningful sites before comparison.
    If not supplied, all instructions are used.
 
    Parameters
    ----------
    individual : Individual
        An instance from individual.py.
    output_registers : list of (MemoryType, int), optional
        Output registers used for intron removal.
 
    Returns
    -------
    Component
        A tuple of instruction strings that can be stored in a set or dict.
    """
    if output_registers is not None:
        prog = individual.get_effective_program(output_registers)
    else:
        prog = individual.program
    return tuple(_instruction_to_str(instr) for instr in prog.instructions)
 
 
# ===========================================================================
# 2. Pure metric functions
#    These operate on already-filtered sets of Components and are fully
#    independent of the rest of the codebase.
# ===========================================================================
 
def calculate_change(
    prev_persistent: Sequence[Component],
    curr_persistent: Sequence[Component],
) -> int:
    """
    Change metric (Equation 1 in the paper).
 
    Counts components in the current timepoint's persistent set that were
    *not* present in the previous timepoint's persistent set.
 
    Parameters
    ----------
    prev_persistent : sequence of Component
        Persistent components from the previous measurement window.
    curr_persistent : sequence of Component
        Persistent components from the current measurement window.
 
    Returns
    -------
    int
        Number of components that changed (appeared for the first time
        relative to the previous window).
    """
    prev_set = set(prev_persistent)
    curr_set = set(curr_persistent)
    return len(curr_set - prev_set)
 
 
def calculate_novelty(
    all_historical: Set[Component],
    curr_persistent: Sequence[Component],
) -> int:
    """
    Novelty metric (Equation 2 in the paper).
 
    Counts components in the current persistent set that have *never*
    appeared in any previous persistent set.  Once a component is counted
    as novel it is added to *all_historical* so it is never double-counted.
 
    Parameters
    ----------
    all_historical : set of Component  [mutated in-place]
        Cumulative set of all components ever seen.  Pass the same set
        object on every call so history accumulates across generations.
    curr_persistent : sequence of Component
        Persistent components from the current measurement window.
 
    Returns
    -------
    int
        Number of genuinely novel components discovered this generation.
    """
    curr_set = set(curr_persistent)
    novel = curr_set - all_historical
    all_historical.update(novel)  # update history in-place
    return len(novel)
 
 
def calculate_ecology(population_frequencies: Sequence[float]) -> float:
    """
    Ecological potential metric (Equation 3 in the paper).
 
    Shannon entropy (base-2) over the relative frequencies of persistent
    genotypes.  Higher values indicate more diverse, evenly distributed
    populations.
 
    Parameters
    ----------
    population_frequencies : sequence of float
        Counts or proportions for each distinct component.  Raw counts
        are fine — scipy normalises internally.
 
    Returns
    -------
    float
        Shannon entropy H = -Σ P(c) log₂ P(c).
    """
    return float(entropy(population_frequencies, base=2))
 
 
def calculate_complexity(
    individuals,
    output_registers=None,
) -> int:
    """
    Complexity metric (Section 3.2.3 of the paper).
 
    The complexity of a population at a given timepoint is the *maximum*
    number of meaningful (effective) instructions across all individuals
    that passed the persistence filter.
 
    The paper recommends using information-theoretic site counting; here
    we approximate this by counting effective (non-intron) instructions,
    which is the natural analogue for LGP genomes.
 
    Parameters
    ----------
    individuals : iterable of Individual
        The individuals that passed the persistence filter.
    output_registers : list of (MemoryType, int), optional
        Required to strip introns.  If None, total instruction count is used.
 
    Returns
    -------
    int
        Maximum effective instruction count across the supplied individuals.
        Returns 0 if the iterable is empty.
    """
    max_complexity = 0
    for ind in individuals:
        if output_registers is not None:
            length = ind.get_effective_length(output_registers)
        else:
            length = len(ind.program)
        if length > max_complexity:
            max_complexity = length
    return max_complexity
 
 
# ===========================================================================
# 3. Persistence filter
# ===========================================================================
 
@dataclass
class PersistenceFilter:
    """
    Ancestry-based persistence filter (Section 3.1.1 of the paper).

    At generation A, each individual is recorded with its id and
    parent_ids. After `filter_length` generations (A + filter_length),
    we walk the parent_ids chain backward from the *current* population
    to generation A, building the exact set of A-generation ids that
    are ancestors of individuals alive now. Only those individuals
    are "persistent".
    """

    filter_length: int

    # generation -> list of (id, parent_ids, Individual)
    _snapshots: Dict[int, List[Tuple[int, Tuple[int, ...], object]]] = field(
        default_factory=dict, init=False, repr=False
    )

    def snapshot(self, population, generation: int) -> None:
        self._snapshots[generation] = [
            (ind.id, ind.parent_ids, ind) for ind in population.individuals
        ]
        # Drop snapshots we'll never need for a future query.
        cutoff = generation - self.filter_length
        for g in list(self._snapshots.keys()):
            if g < cutoff:
                del self._snapshots[g]

    def get_persistent(self, generation: int, population) -> List:
        target_gen = generation - self.filter_length
        if target_gen < 0 or target_gen not in self._snapshots:
            return []

        # Start from the ids alive right now and walk backward.
        ancestor_ids: Set[int] = {ind.id for ind in population.individuals}

        for g in range(generation, target_gen, -1):
            if g not in self._snapshots or (g - 1) not in self._snapshots:
                return []  # broken chain — shouldn't happen if record() is called every gen

            parent_ids_by_id = {iid: pids for iid, pids, _ in self._snapshots[g]}
            prev_ids = {iid for iid, _, _ in self._snapshots[g - 1]}

            next_ids: Set[int] = set()
            for iid in ancestor_ids:
                if iid in prev_ids:
                    # Same id existed last generation -> elite/carry-over,
                    # it IS its own ancestor at g-1.
                    next_ids.add(iid)
                else:
                    # Freshly created this generation -> trace to its
                    # recorded parents from g-1.
                    next_ids.update(parent_ids_by_id.get(iid, ()))
            ancestor_ids = next_ids

        return [ind for iid, _, ind in self._snapshots[target_gen] if iid in ancestor_ids] 
 
# ===========================================================================
# 4. MODESTracker — stateful per-generation recorder
# ===========================================================================
 
@dataclass
class MODESRecord:
    """One row of MODES measurements."""
    generation: int
    change: int
    novelty: int
    ecology: float
    complexity: int
    n_persistent: int  # how many individuals passed the filter
 
 
class MODESTracker:
    """
    Stateful tracker that computes all four MODES metrics each generation
    and accumulates a history you can inspect or plot.
 
    Parameters
    ----------
    filter_length : int
        Passed to PersistenceFilter.  Rule of thumb: use population size.
    output_registers : list of (MemoryType, int), optional
        Output registers for intron removal.  Strongly recommended; without
        it, introns inflate novelty and complexity counts.
 
    Example
    -------
    >>> tracker = MODESTracker(filter_length=pop_size,
    ...                        output_registers=evaluator.output_registers)
    >>> for gen in range(max_gens):
    ...     engine.run_one_generation()
    ...     tracker.record(engine.population, gen)
    >>> df = tracker.to_dataframe()
    """
 
    def __init__(
        self,
        filter_length: int,
        output_registers=None,
    ) -> None:
        self.filter_length = filter_length
        self.output_registers = output_registers
 
        self._persistence_filter = PersistenceFilter(filter_length=filter_length)
 
        # For change metric: persistent components from previous window
        self._prev_persistent_components: List[Component] = []
 
        # For novelty metric: cumulative set of all components ever seen
        self._all_historical_components: Set[Component] = set()
 
        # Accumulated results
        self.history: List[MODESRecord] = []
 
    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
 
    def record(self, population, generation: int) -> MODESRecord:
        """
        Snapshot the population, apply the persistence filter, compute all
        four metrics, and append a MODESRecord to ``self.history``.
 
        Call this once per generation *after* evaluation and replacement.
 
        Parameters
        ----------
        population : Population
            The current population (post-replacement).
        generation : int
            Current generation index (0-based).
 
        Returns
        -------
        MODESRecord
            The record for this generation (also appended to self.history).
        """
        # 1. Record snapshot for future persistence checks
        self._persistence_filter.snapshot(population, generation)
 
        # 2. Retrieve persistent individuals from filter_length gens ago
        persistent_individuals = self._persistence_filter.get_persistent(
            generation, population
        )
 
        # 3. Derive components (effective instruction fingerprints)
        curr_persistent_components: List[Component] = [
            get_component(ind, self.output_registers)
            for ind in persistent_individuals
        ]
 
        # 4. Compute metrics
        change = calculate_change(
            self._prev_persistent_components,
            curr_persistent_components,
        )
 
        novelty = calculate_novelty(
            self._all_historical_components,  # mutated in-place
            curr_persistent_components,
        )
 
        ecology = 0.0
        if curr_persistent_components:
            # Count occurrences of each distinct component for Shannon entropy
            freq: Dict[Component, int] = defaultdict(int)
            for c in curr_persistent_components:
                freq[c] += 1
            ecology = calculate_ecology(list(freq.values()))
 
        complexity = calculate_complexity(
            persistent_individuals,
            self.output_registers,
        )
 
        # 5. Build record
        record = MODESRecord(
            generation=generation,
            change=change,
            novelty=novelty,
            ecology=ecology,
            complexity=complexity,
            n_persistent=len(persistent_individuals),
        )
        self.history.append(record)
 
        # 6. Advance sliding window for change metric
        self._prev_persistent_components = curr_persistent_components
 
        return record
 
    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
 
    def to_dataframe(self):
        """
        Return history as a pandas DataFrame.
        Raises ImportError if pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for to_dataframe(); "
                "install it with: pip install pandas"
            ) from exc
 
        return pd.DataFrame(
            [
                {
                    "generation": r.generation,
                    "change": r.change,
                    "novelty": r.novelty,
                    "ecology": r.ecology,
                    "complexity": r.complexity,
                    "n_persistent": r.n_persistent,
                }
                for r in self.history
            ]
        )
 
    def print_latest(self) -> None:
        """Print a one-line summary of the most recent record."""
        if not self.history:
            print("MODESTracker: no records yet.")
            return
        r = self.history[-1]
        print(
            f"[Gen {r.generation:4d}] MODES — "
            f"change={r.change:3d}  novelty={r.novelty:3d}  "
            f"ecology={r.ecology:.3f}  complexity={r.complexity:3d}  "
            f"(n_persistent={r.n_persistent})"
        )