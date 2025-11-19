## Evaluator Documentation

## Overview

Evaluators are responsible for assessing the fitness of individuals. The system uses **intron-removed (effective) programs** during evaluation to improve performance and accuracy by executing only the instructions that actually affect the output.

## How Evaluators Use Effective Programs

All evaluators automatically use the effective program (intron-removed) when evaluating individuals. This is done by calling `individual.get_effective_program(output_registers)` instead of using `individual.program` directly.

**Key Benefits:**
- **Performance**: Only effective instructions are executed, reducing computation time
- **Accuracy**: Introns don't interfere with program behavior
- **Caching**: The effective program is computed once and cached for subsequent uses

**Example Pattern:**
```python
def _evaluate_episode(self, individual: 'Individual', episode_idx: int) -> float:
    memory = individual.memory.copy()
    # ... load observations ...
    
    # Use effective program instead of full program
    effective_program = individual.get_effective_program(self.output_registers)
    effective_program.execute(memory)
    
    # ... read output and compute fitness ...
```

**Important Notes:**
- The effective program uses the same memory interface - you pass memory to it at execution time
- The effective program doesn't have its own memory - it operates on the memory you provide
- The cache is automatically invalidated when the individual's program changes

## Example Implementations

### `SymbolicRegressionEvaluator`

- Pulls two scalar observations, executes the program, and penalizes absolute error.
- Useful for sanity tests.
- **Note**: Currently uses `individual.program` directly (should be updated to use effective program).

### `CartPoleEvaluator`

- Wraps `gymnasium`'s `CartPole-v1` environment.
- Assumes four scalar observation registers and reads an action from working scalar register `output_register` (default 7).
- Converts the register value into a discrete action (`0` or `1`) and averages episode returns.
- **Uses effective program**: Calls `individual.get_effective_program(self.output_registers).execute(memory)`
- Optional parameters: `episodes`, `max_steps`, `render_mode`.
- Requires `gymnasium`; raises an informative error if the library is missing.

**Example Implementation:**
```python
def _evaluate_episode(self, individual: 'Individual', episode_idx: int) -> float:
    observation, _ = self.env.reset()
    memory = individual.memory.copy()
    total_reward = 0.0

    for _ in range(self.max_steps):
        memory.load_observation({'scalar': observation.tolist()})
        
        # Execute effective program (intron-removed)
        effective_program = individual.get_effective_program(self.output_registers)
        effective_program.execute(memory)

        action_value = memory.read_scalar(self.output_register)
        action = 1 if action_value >= 0.0 else 0
        # ... rest of evaluation ...
```

### `FlappyBirdEvaluator`

- Evaluates policies on FlappyBird using image observations.
- Processes RGB images into matrix observations.
- Reads action from output register.
- **Uses effective program**: Calls `individual.get_effective_program(self.output_registers).execute(memory)`
- Supports multiple image processing strategies (`full_image`, `quantized`).

**Example Implementation:**
```python
def _evaluate_episode(self, individual: 'Individual', episode_idx: int) -> float:
    observation, _ = self.env.reset()
    memory = individual.memory.copy()
    total_reward = 0.0

    for _ in range(self.max_steps):
        # Process image observation
        matrix_observations = self._process_observation(observation)
        memory.load_observation({'matrix': matrix_observations})

        # Execute effective program (intron-removed)
        effective_program = individual.get_effective_program(self.output_registers)
        effective_program.execute(memory)

        # Read action and step environment
        action_value = memory.read_scalar(self.output_register)
        action = 1 if action_value >= 0.0 else 0
        # ... rest of evaluation ...
```

### `FitnessEvaluator` (Base Class)

Base class for all evaluators with common functionality:

**Parameters:**
- `episodes`: Number of episodes to average over
- `rng`: Optional random number generator (for reproducibility)
- `output_registers`: **Required** `List[Tuple[MemoryType, int]]` describing where the program outputs are read. This is used to compute the effective program.

**Important:** All evaluators should specify `output_registers` in their `__init__` method and use `individual.get_effective_program(self.output_registers)` when executing programs.

## Best Practices

1. **Always specify output_registers**: Every evaluator should define which registers contain the program output
2. **Use effective program**: Call `individual.get_effective_program(self.output_registers)` instead of `individual.program`
3. **Pass memory explicitly**: The effective program doesn't have its own memory - pass it at execution time
4. **Cache is automatic**: The effective program is cached per individual/output_registers combination
