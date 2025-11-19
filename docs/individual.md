# Individual Documentation

## Overview

`Individual` is the container for candidate solutions in the system. Each individual bundles:

- A `Program` (sequence of instructions)
- A `MemoryBank` holding evolvable constants and observation registers
- Metadata used by the evolutionary loop (id, fitness, age, lineage)

## Key Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `program` | `Program` | Executable LGP program (full program with all instructions) |
| `memory` | `MemoryBank` | Working/evolutionary registers |
| `fitness` | `Optional[float]` | Cached fitness value |
| `age` | `int` | Generational age |
| `parent_ids` | `Tuple[int, ...]` | Lineage tracking |
| `_effective_program` | `Optional[Program]` | Cached intron-removed program (internal, lazy-computed) |
| `_output_registers` | `Optional[List[Tuple[MemoryType, int]]]` | Cached output registers used for effective program (internal) |

## Factory Method

`Individual.random(...)` constructs fresh candidates:

```python
@classmethod
def random(
    cls,
    instruction_set: InstructionSet,
    memory_config: MemoryConfig,
    program_length: int,
    rng: Optional[np.random.Generator] = None,
    mutate_constants: bool = True,
) -> Individual:
    ...
```

Steps:
1. Draw a random program from the supplied `InstructionSet`.
2. Allocate a new `MemoryBank` using the provided `MemoryConfig`.
3. Optionally call `GeneticOperators.mutate_constants` for a stochastic constant shake-up.

This keeps population initialization concise and makes it easy to spawn reproducible individuals by passing a seeded RNG.

## Evaluation Helpers

- `evaluate(evaluator)` defers to any `FitnessEvaluator` implementation.
- `invalidate_fitness()` clears the cached score and effective program cache when the individual mutates.

## Intron Removal and Effective Program

### `get_effective_program(output_registers: List[Tuple[MemoryType, int]]) -> Program`

Get the intron-removed (effective) program, computing lazily if needed.

**Parameters:**
- `output_registers`: List of `(MemoryType, index)` tuples specifying output registers

**Returns:**
- `Program`: Program with only effective instructions (introns removed)

**Behavior:**
- **Lazy computation**: The effective program is computed on first call and cached
- **Cache invalidation**: Cache is cleared when `invalidate_fitness()` is called
- **Efficient**: Subsequent calls with the same `output_registers` return the cached program
- **Task-specific**: Different output registers result in different effective programs

**Example:**
```python
# Get effective program for a specific output register
output_regs = [(MemoryType.SCALAR, 9)]
effective = individual.get_effective_program(output_regs)

# Execute the effective program (faster - no introns)
memory = individual.memory.copy()
memory.load_observation({'scalar': [1.0, 2.0]})
effective.execute(memory)
```

**Important:** The effective program uses the same memory interface as the original program. It doesn't have its own memory - you pass memory to it at execution time.

### `get_effective_length(output_registers: List[Tuple[MemoryType, int]]) -> int`

Get the number of effective instructions (intron-removed program length).

**Parameters:**
- `output_registers`: List of output register specifications

**Returns:**
- `int`: Number of effective instructions

**Use Cases:**
- Fitness metrics (smaller effective programs are better)
- Bloat control
- Program analysis

**Example:**
```python
output_regs = [(MemoryType.SCALAR, 9)]
effective_len = individual.get_effective_length(output_regs)
total_len = len(individual.program)
print(f"Effective: {effective_len}/{total_len}")
```

### `get_intron_ratio(output_registers: List[Tuple[MemoryType, int]]) -> float`

Get the ratio of intron instructions to total instructions.

**Parameters:**
- `output_registers`: List of output register specifications

**Returns:**
- `float`: Ratio in [0, 1] where 0 = no introns, 1 = all introns

**Example:**
```python
output_regs = [(MemoryType.SCALAR, 9)]
ratio = individual.get_intron_ratio(output_regs)
print(f"Intron ratio: {ratio:.1%}")  # e.g., "40.0%"
```

## Copying and Offspring

- `copy(new_id: bool = True)` deep-copies both program and memory.
- `create_offspring(parent_ids)` is a convenience wrapper used by the evolution engine to reset lineage, age, and fitness.

## Usage Example

```python
from Individual import Individual
from instruction_set import InstructionSet
from memory_system import MemoryConfig

cfg = MemoryConfig(...)
instr_set = InstructionSet(operations, template_memory)
ind = Individual.random(instr_set, cfg, program_length=8, rng=rng)
print(ind.program.to_string())
```
