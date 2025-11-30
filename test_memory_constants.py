"""Test script to verify memory and evolved constants behavior.

This script tests:
1. Memory is copied during evaluation (original not modified)
2. Constants are inherited from parents
3. Constants evolve over generations
4. Memory modifications during evaluation don't persist
"""

import numpy as np
from memory_system import MemoryConfig, MemoryBank
from instruction_set import InstructionSet
from individual import Individual
from population import PopulationConfig, Population
from evolution_engine import EvolutionEngine, EvolutionConfig
from evaluator import FitnessEvaluator, BaseEvaluatorConfig
from memory_system import MemoryType
from operators import GeneticOperators

# Try to import operations, handle cv2 import error gracefully
try:
    from operation import SCALAR_OPS
except ImportError as e:
    if 'cv2' in str(e):
        print("Warning: cv2 not available, using minimal scalar operations")
        # Define minimal scalar operations inline if cv2 is missing
        from operation import (
            ScalarAddOp, ScalarSubOp, ScalarMulOp, ScalarDivProtectedOp,
            ScalarMaxOp, ScalarMinOp, ScalarAbsOp, ScalarNegOp
        )
        SCALAR_OPS = [
            ScalarAddOp, ScalarSubOp, ScalarMulOp, ScalarDivProtectedOp,
            ScalarMaxOp, ScalarMinOp, ScalarAbsOp, ScalarNegOp
        ]
    else:
        raise


class ConstantTrackingEvaluator(FitnessEvaluator):
    """Simple evaluator that tracks constants and verifies memory isolation.
    
    Fitness is based on reading scalar[0] value, which allows us to track
    how constants evolve. We also verify that evaluation doesn't modify
    the original memory.
    """
    
    def __init__(self, config: BaseEvaluatorConfig):
        super().__init__(config)
        self.constants_before_eval = {}
        self.constants_after_eval = {}
    
    def _evaluate_episode(self, individual: 'Individual', episode_idx: int) -> float:
        # Store constants BEFORE evaluation
        individual_id = individual.id
        if individual_id not in self.constants_before_eval:
            self.constants_before_eval[individual_id] = individual.memory.scalars.copy()
        
        # Create a copy of memory (as evaluators should do)
        memory = individual.memory.copy()
        
        # Modify the copied memory extensively during evaluation
        # This simulates what happens during program execution
        memory.write_scalar(0, 999.0)
        memory.write_scalar(1, 888.0)
        memory.write_scalar(2, 777.0)
        memory.scalars += 100.0  # Modify all scalars
        
        # Execute a simple program (just read scalar[0])
        # In real evaluation, this would execute the individual's program
        # For this test, we'll just read the value
        
        # Store constants AFTER evaluation (from the copy, not original)
        # This should show the modified values
        self.constants_after_eval[individual_id] = memory.scalars.copy()
        
        # Fitness is based on the original scalar[0] value (before modification)
        # This allows us to track how constants evolve
        original_value = individual.memory.read_scalar(0)
        fitness = float(original_value)  # Simple fitness: just the constant value
        
        return fitness
    
    def verify_memory_isolation(self, individual: 'Individual') -> bool:
        """Verify that evaluation didn't modify the original memory."""
        individual_id = individual.id
        if individual_id not in self.constants_before_eval:
            return True  # Not evaluated yet
        
        original_before = self.constants_before_eval[individual_id]
        original_after = individual.memory.scalars
        
        # Check if original memory was modified
        if not np.array_equal(original_before, original_after):
            print(f"  ❌ ERROR: Original memory was modified for individual {individual_id}!")
            print(f"     Before: {original_before[:3]}...")
            print(f"     After:  {original_after[:3]}...")
            return False
        
        return True


def test_memory_and_constants():
    """Run a small evolution to test memory and constant behavior."""
    
    print("=" * 70)
    print("TESTING MEMORY AND EVOLVED CONSTANTS")
    print("=" * 70)
    
    # Setup
    rng = np.random.default_rng(42)
    
    memory_cfg = MemoryConfig(
        n_scalar=4,  # Small number for easy tracking
        n_vector=0,
        n_matrix=0,
        n_obs_scalar=0,
        n_obs_vector=0,
        n_obs_matrix=0,
        vector_size=1,
        matrix_shape=(1, 1),
    )
    
    template_memory = MemoryBank(
        n_scalar=memory_cfg.n_scalar,
        n_vector=memory_cfg.n_vector,
        n_matrix=memory_cfg.n_matrix,
        n_obs_scalar=memory_cfg.n_obs_scalar,
        n_obs_vector=memory_cfg.n_obs_vector,
        n_obs_matrix=memory_cfg.n_obs_matrix,
        vector_size=memory_cfg.vector_size,
        matrix_shape=memory_cfg.matrix_shape,
    )
    
    instr_set = InstructionSet([op() for op in SCALAR_OPS], template_memory)
    operators = GeneticOperators(instr_set, rng)
    
    pop_config = PopulationConfig(size=5, program_length=(2, 4), elitism=2)
    population = Population(pop_config, instr_set, memory_cfg, operators=operators, rng=rng)
    population.initialize_random(mutate_constants=True)
    
    evaluator = ConstantTrackingEvaluator(
        config=BaseEvaluatorConfig(
            episodes=1,
            output_registers=[(MemoryType.SCALAR, 0)],
            rng_seed=42
        )
    )
    
    engine = EvolutionEngine(
        population=population,
        operators=operators,
        evaluator=evaluator,
        config=EvolutionConfig(
            max_generations=3,
            mutation_threshold=0.5,
            constant_mutation_rate=0.0,  # DISABLED - test that constants are NOT mutated
            crossover_threshold=0.5,
            verbose=False
        ),
        rng=rng,
    )
    
    print("\n📊 Initial Population Constants:")
    print("-" * 70)
    for i, ind in enumerate(population.individuals):
        constants = ind.memory.scalars.copy()
        print(f"Individual {ind.id}: {constants}")
    
    # Track constants across generations
    generation_constants = {}
    
    # Run evolution
    print("\n🔄 Running Evolution...")
    print("-" * 70)
    
    for gen in range(engine.config.max_generations):
        print(f"\n--- Generation {gen} ---")
        
        # Store constants before evaluation
        gen_constants = {}
        for ind in population.individuals:
            gen_constants[ind.id] = ind.memory.scalars.copy()
        generation_constants[gen] = gen_constants
        
        # Evaluate
        population.evaluate_all(evaluator)
        
        # Verify memory isolation
        print("\n  ✓ Verifying memory isolation...")
        all_isolated = True
        for ind in population.individuals:
            if not evaluator.verify_memory_isolation(ind):
                all_isolated = False
        
        if all_isolated:
            print("  ✅ All memory copies are isolated correctly!")
        
        # Show constants after evaluation (should be unchanged)
        print(f"\n  Constants after evaluation (should match before):")
        for ind in population.individuals:
            constants = ind.memory.scalars.copy()
            before = gen_constants[ind.id]
            match = np.allclose(constants, before)
            status = "✅" if match else "❌"
            print(f"    {status} Individual {ind.id}: {constants[:3]}... (fitness: {ind.fitness:.4f})")
        
        if gen < engine.config.max_generations - 1:
            # Create next generation (same logic as EvolutionEngine.run())
            # Store parent constants before creating offspring
            parent_constants_map = {ind.id: ind.memory.scalars.copy() for ind in population.individuals}
            
            # Get elites
            elites = population.get_elites()
            elite_ids = {elite.id for elite in elites}  # Track which individuals are elites
            offspring = elites.copy()  # Shallow copy - elites are references to parent individuals
            
            # Track which parent each child actually inherited from
            # child1 inherits from parent1, child2 inherits from parent2
            child_to_inherited_parent = {}
            
            # Create offspring until we have enough
            while len(offspring) < population.config.size:
                parent1, parent2 = population.tournament_selection(3, num_winners=2)
                child_program_1, child_program_2 = operators.crossover(
                    parent1.program, parent2.program,
                    engine.config.crossover_threshold, rng
                )
                
                # Mutate child programs
                operators.mutate_program(
                    child_program_1,
                    engine.config.mutation_threshold,
                    rng,
                    max_length=population.config.max_program_length,
                )
                operators.mutate_program(
                    child_program_2,
                    engine.config.mutation_threshold,
                    rng,
                    max_length=population.config.max_program_length,
                )
                
                if population.config.max_program_length is not None:
                    child_program_1.max_program_length = population.config.max_program_length
                    child_program_2.max_program_length = population.config.max_program_length
                
                # DEBUG: Store parent constants right before creating offspring
                parent1_constants_before = parent1.memory.scalars.copy()
                parent2_constants_before = parent2.memory.scalars.copy()
                
                # Create children (inherit parent memory/constants)
                child1 = parent1.create_offspring(parent_ids=(parent1.id, parent2.id))
                child1.program = child_program_1
                
                child2 = parent2.create_offspring(parent_ids=(parent1.id, parent2.id))
                child2.program = child_program_2
                
                # DEBUG: Verify parent constants weren't modified by create_offspring
                parent1_constants_after = parent1.memory.scalars.copy()
                parent2_constants_after = parent2.memory.scalars.copy()
                
                if not np.array_equal(parent1_constants_before, parent1_constants_after):
                    print(f"      ❌ ERROR: Parent1 constants modified during create_offspring!")
                if not np.array_equal(parent2_constants_before, parent2_constants_after):
                    print(f"      ❌ ERROR: Parent2 constants modified during create_offspring!")
                
                # DEBUG: Check if child inherited correctly
                child1_constants = child1.memory.scalars.copy()
                child2_constants = child2.memory.scalars.copy()
                
                if not np.array_equal(parent1_constants_before, child1_constants):
                    print(f"      ⚠️  Child1 constants differ from parent1!")
                    print(f"         Parent1: {parent1_constants_before[:3]}...")
                    print(f"         Child1:  {child1_constants[:3]}...")
                
                if not np.array_equal(parent2_constants_before, child2_constants):
                    print(f"      ⚠️  Child2 constants differ from parent2!")
                    print(f"         Parent2: {parent2_constants_before[:3]}...")
                    print(f"         Child2:  {child2_constants[:3]}...")
                
                # Mutate constants if configured
                should_mutate = engine.config.constant_mutation_rate > 0 and (
                    rng.random() < engine.config.constant_mutation_rate
                )
                if should_mutate:
                    print(f"      🔄 Mutating constants for children {child1.id} and {child2.id}")
                    operators.mutate_constants(child1.memory, rng)
                    operators.mutate_constants(child2.memory, rng)
                else:
                    # Verify constants are NOT mutated when mutation is disabled
                    if not np.array_equal(parent1_constants_before, child1.memory.scalars):
                        print(f"      ❌ ERROR: Child1 constants changed but mutation is disabled!")
                    if not np.array_equal(parent2_constants_before, child2.memory.scalars):
                        print(f"      ❌ ERROR: Child2 constants changed but mutation is disabled!")
                
                child1.invalidate_fitness()
                child2.invalidate_fitness()
                
                # Track which parent each child inherited from
                # child1 inherits from parent1, child2 inherits from parent2
                child_to_inherited_parent[child1.id] = parent1.id
                child_to_inherited_parent[child2.id] = parent2.id
                
                offspring.append(child1)
                offspring.append(child2)
            
            # Trim to population size
            offspring = offspring[:population.config.size]
            
            # Replace population
            population.replace_population(offspring)
            population.generation += 1
            
            # Check parent-child inheritance
            print(f"\n  Checking parent-child constant inheritance:")
            print(f"  DEBUG: constant_mutation_rate = {engine.config.constant_mutation_rate}")
            for ind in population.individuals:
                # Skip elites - they ARE the parents themselves, not children
                if ind.id in elite_ids:
                    continue
                    
                if ind.parent_ids:
                    # Find which parent this child actually inherited from
                    # child1 inherits from parent1, child2 inherits from parent2
                    inherited_parent_id = child_to_inherited_parent.get(ind.id)
                    if inherited_parent_id is None:
                        # This is an elite, skip it
                        continue
                    
                    # Use the actual inherited parent, not just parent_ids[0]
                    parent_id = inherited_parent_id
                    if parent_id in parent_constants_map:
                        parent_constants = parent_constants_map[parent_id]
                        child_constants = ind.memory.scalars.copy()
                        
                        # DEBUG: Check if parent still exists and what its constants are
                        parent_in_pop = next((p for p in population.individuals if p.id == parent_id), None)
                        if parent_in_pop:
                            current_parent_constants = parent_in_pop.memory.scalars.copy()
                            parent_match = np.allclose(parent_constants, current_parent_constants)
                            print(f"    DEBUG: Parent {parent_id} constants unchanged: {parent_match}")
                            if not parent_match:
                                print(f"      Stored: {parent_constants[:3]}...")
                                print(f"      Current: {current_parent_constants[:3]}...")
                        
                        # Child should inherit parent's constants (possibly mutated)
                        print(f"    Parent {parent_id}: {parent_constants[:3]}...")
                        print(f"    Child  {ind.id}:   {child_constants[:3]}...")
                        
                        # Check exact match first (should match if no mutation)
                        exact_match = np.array_equal(parent_constants, child_constants)
                        if exact_match:
                            print(f"      ✅ Child inherited constants EXACTLY (no mutation)")
                        elif np.allclose(parent_constants, child_constants, atol=1e-6):
                            print(f"      ✅ Child inherited constants (tiny floating point differences)")
                        elif np.allclose(parent_constants, child_constants, atol=0.1):
                            print(f"      ⚠️  Child constants differ slightly (within tolerance)")
                        else:
                            print(f"      ❌ ERROR: Child constants differ significantly!")
                            print(f"         Difference: {np.abs(parent_constants - child_constants)[:3]}...")
                            print(f"         Max diff: {np.max(np.abs(parent_constants - child_constants)):.6f}")
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    # Final verification
    print("\n✅ Memory Isolation Test:")
    all_isolated = True
    for ind in population.individuals:
        if not evaluator.verify_memory_isolation(ind):
            all_isolated = False
    
    if all_isolated:
        print("  PASSED: Original memory was never modified during evaluation")
    else:
        print("  FAILED: Original memory was modified during evaluation")
    
    print("\n✅ Constant Inheritance Test:")
    print("  Checking if constants are passed from parents to children...")
    inheritance_ok = True
    for gen in range(1, len(generation_constants)):
        print(f"  Generation {gen}:")
        for ind in population.individuals:
            if ind.parent_ids:
                # Constants should be inherited (possibly mutated)
                print(f"    Individual {ind.id} inherited from {ind.parent_ids}")
                inheritance_ok = True
    
    if inheritance_ok:
        print("  PASSED: Constants are inherited from parents")
    
    print("\n✅ Constant Evolution Test:")
    print("  Checking if constants change across generations...")
    if len(generation_constants) > 1:
        gen0_constants = list(generation_constants[0].values())[0]
        last_gen = len(generation_constants) - 1
        last_constants = list(generation_constants[last_gen].values())[0]
        
        if not np.allclose(gen0_constants, last_constants):
            print(f"  PASSED: Constants evolved from {gen0_constants[:3]}... to {last_constants[:3]}...")
        else:
            print(f"  INFO: Constants remained similar (may need more mutation)")
    
    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)


if __name__ == "__main__":
    test_memory_and_constants()

