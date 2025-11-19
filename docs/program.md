# Program Documentation

## Overview

The `program.py` module defines the `Program` class, which represents a complete Linear Genetic Programming (LGP) program as a sequence of instructions. The `Program` class provides execution, analysis, and optimization capabilities including intron (dead code) removal.

## Class: `Program`

### Class Responsibilities

1. **Program Execution**: Execute a sequence of instructions on a memory bank
2. **Intron Detection**: Identify instructions that don't affect the output (dead code)
3. **Intron Removal**: Create optimized programs with introns removed
4. **Dependency Analysis**: Analyze register dependencies between instructions
5. **Program Analysis**: Provide statistics and metrics about program effectiveness

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `instructions` | `List[Instruction]` | List of instructions in the program |
| `_effective_instructions` | `Set[int]` | Cached set of effective instruction indices (lazy) |

---

## Initialization

```python
Program(instructions: List[Instruction])
```

**Parameters:**
- `instructions`: List of `Instruction` objects in execution order

**Example:**
```python
from program import Program
from instruction import Instruction
from operation import ScalarAddOp, ScalarMulOp
from memory_system import MemoryType

# Create instructions
instrs = [
    Instruction(ScalarAddOp(), MemoryType.SCALAR, 0,
               [MemoryType.SCALAR, MemoryType.SCALAR], [-1, -2]),
    Instruction(ScalarMulOp(), MemoryType.SCALAR, 1,
               [MemoryType.SCALAR, MemoryType.SCALAR], [0, 0]),
    Instruction(ScalarAddOp(), MemoryType.SCALAR, 9,
               [MemoryType.SCALAR, MemoryType.SCALAR], [1, 1])
]

# Create program
program = Program(instrs)
```

---

## Core Methods

### `execute(memory: MemoryBank) -> None`

Execute the program on a memory bank.

**Parameters:**
- `memory`: The MemoryBank to execute on

**Behavior:**
- Executes instructions in sequence
- Each instruction reads from and writes to the memory bank
- Execution order is preserved

**Example:**
```python
from memory_system import MemoryBank

memory = MemoryBank(
    n_scalar=10, n_vector=0, n_matrix=0,
    n_obs_scalar=2, n_obs_vector=0, n_obs_matrix=0,
    vector_size=0, matrix_shape=(0, 0)
)

# Load observations
memory.load_observation({'scalar': [2.0, 3.0]})

# Execute program
program.execute(memory)

# Read output from register 9
output = memory.read_scalar(9)
```

---

### `copy() -> Program`

Create a deep copy of the program.

**Returns:**
- `Program`: A new Program instance with copied instructions

**Note:** Operations are immutable and shared between copies.

**Example:**
```python
program_copy = program.copy()
# Modifications to program_copy won't affect program
```

---

## Intron Removal System

### Overview

**Introns** are instructions that don't affect the program's output. They are "dead code" that can be safely removed without changing program behavior.

The intron removal system uses a **simplified backward pass algorithm**:
1. Start with output registers in a set
2. Traverse instructions from back to front
3. If an instruction writes to a register in the set, mark it as effective
4. Remove the written register from the set and add the instruction's source registers
5. Continue until all dependencies are traced

This algorithm is more efficient than previous implementations, requiring only a single backward pass with simple set operations.

### Key Methods

#### `intron_removal(output_registers: List[Tuple[MemoryType, int]]) -> None`

Remove introns **in-place** by modifying the program's instruction list.

**Parameters:**
- `output_registers`: List of `(MemoryType, index)` tuples specifying output registers

**Behavior:**
- **Modifies the program in-place** - the original instruction list is replaced
- Uses a simple backward pass algorithm
- Maintains instruction order (only removes introns)
- More efficient than previous implementations

**⚠️ Warning:** This method modifies the program. Use `remove_introns_create_copy()` if you need to preserve the original.

**Example:**
```python
# Create a copy first if you want to preserve the original
program_copy = program.copy()
program_copy.intron_removal([(MemoryType.SCALAR, 9)])
# program_copy now has introns removed
```

#### `remove_introns_create_copy(output_registers: List[Tuple[MemoryType, int]]) -> Program`

Create a new program with introns removed (non-destructive).

**Parameters:**
- `output_registers`: List of output register specifications

**Returns:**
- `Program`: New Program with only effective instructions (maintains order)

**Example:**
```python
compact_program = program.remove_introns_create_copy([(MemoryType.SCALAR, 9)])
print(len(program))  # 10 (original unchanged)
print(len(compact_program))  # 5 (if 5 were introns)
```

---

## Internal Methods

### `_build_write_map() -> Dict[Tuple[MemoryType, int], List[int]]`

Build a map from registers to instructions that write to them.

**Returns:**
- `Dict`: Maps `(register_type, register_index)` to list of instruction indices

**Purpose:**
- Used for dependency analysis
- Pre-computed once for efficiency

**Example:**
```python
write_map = program._build_write_map()
# { (MemoryType.SCALAR, 0): [0, 2, 5],
#   (MemoryType.SCALAR, 1): [1],
#   ... }
```

### `_get_last_writer(register: Tuple[MemoryType, int], before_idx: int, write_map: Dict) -> int`

Find the last instruction that wrote to a register before a given index.

**Parameters:**
- `register`: `(MemoryType, index)` tuple
- `before_idx`: Only consider instructions before this index
- `write_map`: Pre-computed write map

**Returns:**
- `int`: Instruction index, or -1 if no writer found

**Purpose:**
- Used in dependency analysis to find which instruction produced a value
- Critical for handling overwritten registers correctly

---

## Pretty Printing

### `to_string(output_registers: List[Tuple[MemoryType, int]] = None, show_introns: bool = True) -> str`

Pretty print the program, optionally marking introns.

**Parameters:**
- `output_registers`: If provided, compute and show introns
- `show_introns`: If True, mark intron instructions with 'X'

**Returns:**
- `str`: Formatted string representation

**Format:**
```
  0: instruction1
  1: instruction2
X 2: instruction3  (if intron)
  3: instruction4

Effective: 3/4 (75.0%)
```

**Example:**
```python
output = program.to_string(output_regs, show_introns=True)
print(output)
```

---

## Special Methods

### `__len__() -> int`

Return the number of instructions in the program.

**Example:**
```python
print(len(program))  # 10
```

### `__repr__() -> str`

Return a string representation of the program.

**Format:**
```
Program(N instructions)
```

**Example:**
```python
print(program)  # Program(10 instructions)
```

---

## Usage Examples

### Basic Program Execution

```python
from program import Program
from instruction import Instruction
from operation import ScalarAddOp, ScalarMulOp
from memory_system import MemoryBank, MemoryType

# Create program
instructions = [
    Instruction(ScalarAddOp(), MemoryType.SCALAR, 0,
               [MemoryType.SCALAR, MemoryType.SCALAR], [-1, -2]),
    Instruction(ScalarMulOp(), MemoryType.SCALAR, 1,
               [MemoryType.SCALAR, MemoryType.SCALAR], [0, 0]),
    Instruction(ScalarAddOp(), MemoryType.SCALAR, 9,
               [MemoryType.SCALAR, MemoryType.SCALAR], [1, 1])
]

program = Program(instructions)

# Execute
memory = MemoryBank(...)
memory.load_observation({'scalar': [2.0, 3.0]})
program.execute(memory)

# Read output
output = memory.read_scalar(9)
```

### Intron Removal

```python
# Create program with introns
instructions = [
    # Effective chain
    Instruction(ScalarAddOp(), MemoryType.SCALAR, 0,
               [MemoryType.SCALAR, MemoryType.SCALAR], [-1, -2]),
    Instruction(ScalarMulOp(), MemoryType.SCALAR, 1,
               [MemoryType.SCALAR, MemoryType.SCALAR], [0, 0]),
    Instruction(ScalarAddOp(), MemoryType.SCALAR, 9,
               [MemoryType.SCALAR, MemoryType.SCALAR], [1, 1]),
    # Introns (don't affect scalar[9])
    Instruction(ScalarSubOp(), MemoryType.SCALAR, 2,
               [MemoryType.SCALAR, MemoryType.SCALAR], [0, 1]),
    Instruction(ScalarMulOp(), MemoryType.SCALAR, 3,
               [MemoryType.SCALAR, MemoryType.SCALAR], [2, 2])
]

program = Program(instructions)
output_regs = [(MemoryType.SCALAR, 9)]

# Remove introns (non-destructive - creates a copy)
compact = program.remove_introns_create_copy(output_regs)
print(len(program))  # 5 (original unchanged)
print(len(compact))  # 3 (introns removed)

# Or remove in-place (modifies the program)
program_copy = program.copy()
program_copy.intron_removal(output_regs)
print(len(program_copy))  # 3
```

### Program Analysis

**Note:** For program analysis metrics like effective length and intron ratio, use the `Individual` class methods instead, which provide cached access to the effective program. See [Individual Documentation](individual.md) for details.

---

## Design Notes

### Intron Removal Algorithm

The intron removal uses a **simplified backward pass algorithm** (based on Brameier & Banzhaf, 2001):

**Algorithm:**
1. Initialize a set with output registers
2. Traverse instructions from last to first
3. For each instruction:
   - If it writes to a register in the set:
     - Mark instruction as effective
     - Remove the written register from the set
     - Add all source registers to the set
4. Reverse the marked instructions to maintain order

**Key Advantages:**
- **Single pass**: Only requires one backward traversal
- **Simple**: Uses basic set operations, no complex data structures
- **Efficient**: O(n) time complexity where n is the number of instructions
- **Correct**: Naturally handles overwritten registers (last writer is found first)

**Example:**
```
Instruction 0: scalar[0] = obs[-1] + obs[-2]  (effective - writes to needed register)
Instruction 1: scalar[1] = scalar[0] * scalar[0]  (effective - writes to needed register)
Instruction 2: scalar[0] = scalar[1] - scalar[1]  (intron - overwrites [0] but [0] not needed)
Instruction 3: scalar[9] = scalar[1] + scalar[1]  (effective - output register)
```

When processing backwards:
- Start with `{scalar[9]}` in the set
- Instruction 3 writes to `scalar[9]` → mark as effective, add `{scalar[1]}`
- Instruction 2 writes to `scalar[0]` → not in set, skip
- Instruction 1 writes to `scalar[1]` → mark as effective, add `{scalar[0]}`
- Instruction 0 writes to `scalar[0]` → mark as effective

**Multiple Output Registers:**
The system supports multiple output registers. All dependencies leading to any output register are traced, making all relevant instructions effective.

**Overwritten Registers:**
The algorithm correctly handles registers that are overwritten multiple times. By processing backwards, the last writer (which is encountered first) is naturally selected.

---

## Notes

- Instructions are executed in order
- Intron removal maintains instruction order in the output
- The `_effective_instructions` cache is lazy (computed on first use)
- Operations are immutable and shared between program copies
- All dependency analysis is done statically (no execution required)

