
from numpy import insert
from instruction import Instruction
from typing import List, Tuple, Dict, Set
from collections import defaultdict

from memory_system import MemoryBank, MemoryType

class Program:
    def __init__(self, instructions: List[Instruction], max_program_length = 256) -> None:
        self.instructions = instructions
        self._effective_instructions = None
        self.max_program_length = max_program_length
    def execute(self, memory:MemoryBank):
        for instruction in self.instructions:
            instruction.execute(memory)

    def copy(self) -> 'Program':
        """Deep copy of program"""
        # Need to copy instructions too (they contain operation references)
        return Program([
            Instruction(
                operation=instr.operation,  # Operations are immutable, can share
                dest_type=instr.dest_type,
                dest_index=instr.dest_index,
                source_types=instr.source_types.copy(),
                source_indices=instr.source_indices.copy()
            )
            for instr in self.instructions
        ], max_program_length=self.max_program_length)
    
    def __len__(self) -> int:
        return len(self.instructions)
    
    def __repr__(self) -> str:
        return f"Program({len(self.instructions)} instructions)"

   
    
    # from M. Brameier and W. Banzhaf, "A comparison of linear genetic programming and neural networks in medical data mining,"
    #  in IEEE Transactions on Evolutionary Computation, vol. 5, no. 1, pp. 17-26, Feb 2001, doi: 10.1109/4235.910462.
    def intron_removal( self, output_registers: List[Tuple[MemoryType, int]]):
        output_set = set(output_registers )
        marked_set = []

        for i in range(len(self.instructions) -1, -1, -1): # go through instructions from back 
            instr = self.instructions[i]
            dest = (instr.dest_type, instr.dest_index)
            if dest in output_set: # mark instruction 
                marked_set.append(instr)
                output_set.discard(dest)
                for src_type, src_index in zip(instr.source_types, instr.source_indices):
                    operand = (src_type, src_index)
                    if operand not in output_set:
                        output_set.add(operand)
        marked_set.reverse()
        self.instructions = marked_set
        



    def remove_introns_create_copy(self, output_registers: List[Tuple[MemoryType, int]]) -> 'Program':
        """Create a new program with introns removed (non-destructive)."""
        new_program = self.copy()
        new_program.intron_removal(output_registers)
        return new_program
    
    

    
    # ==================== PRETTY PRINTING ====================
    
    def to_string(self, 
                  output_registers: List[Tuple[MemoryType, int]] = None,
                  show_introns: bool = True) -> str:
        """
        Pretty print the program, optionally marking introns.
        
        Args:
            output_registers: If provided, compute and show introns
            show_introns: If True, mark intron instructions
        
        Returns:
            Formatted string representation
        """
        if output_registers is None or not show_introns:
            # Simple listing
            lines = []
            for idx, instr in enumerate(self.instructions):
                lines.append(f"{idx:3d}: {instr}")
            return "\n".join(lines)
        
        # Show with intron marking
        effective = self.find_effective_instructions(output_registers)
        lines = []
        for idx, instr in enumerate(self.instructions):
            marker = " " if idx in effective else "X"
            lines.append(f"{marker} {idx:3d}: {instr}")
        
        lines.append(f"\nEffective: {len(effective)}/{len(self.instructions)} "
                    f"({len(effective)/len(self.instructions)*100:.1f}%)")
        
        return "\n".join(lines)

class MatrixProgram:
    """ MATRIX REPRESENTATIONN OF THIS"""
    pass 


# ==================== TEST CODE FOR INTRON_REMOVAL ====================

if __name__ == "__main__":
    from operation import ScalarAddOp, ScalarMulOp, ScalarSubOp, ScalarDivProtectedOp
    
    print("="*70)
    print("TESTING INTRON_REMOVAL METHOD")
    print("="*70)
    
    # Test 1: Simple case - all instructions are effective
    print("\n--- Test 1: All instructions effective ---")
    instructions1 = [
        Instruction(ScalarAddOp(), MemoryType.SCALAR, 0, 
                   [MemoryType.SCALAR, MemoryType.SCALAR], [-1, -2]),  # 0: r0 = obs[-1] + obs[-2]
        Instruction(ScalarMulOp(), MemoryType.SCALAR, 1, 
                   [MemoryType.SCALAR, MemoryType.SCALAR], [0, 0]),    # 1: r1 = r0 * r0
        Instruction(ScalarAddOp(), MemoryType.SCALAR, 9, 
                   [MemoryType.SCALAR, MemoryType.SCALAR], [1, 1]),     # 2: r9 = r1 + r1 (output)
    ]
    prog1 = Program(instructions1.copy())
    print(f"Before: {len(prog1)} instructions")
    for i, instr in enumerate(prog1.instructions):
        print(f"  {i}: r{instr.dest_index} = ...")
    
    prog1.intron_removal([(MemoryType.SCALAR, 9)])
    print(f"After: {len(prog1)} instructions")
    for i, instr in enumerate(prog1.instructions):
        print(f"  {i}: r{instr.dest_index} = ...")
    assert len(prog1) == 3, f"Expected 3 instructions, got {len(prog1)}"
    print("✅ Test 1 passed!")
    
    # Test 2: Case with introns - some instructions don't affect output
    print("\n--- Test 2: Program with introns ---")
    instructions2 = [
        Instruction(ScalarAddOp(), MemoryType.SCALAR, 0, 
                   [MemoryType.SCALAR, MemoryType.SCALAR], [-1, -2]),    # 0: r0 = obs[-1] + obs[-2] (effective)
        Instruction(ScalarMulOp(), MemoryType.SCALAR, 1, 
                   [MemoryType.SCALAR, MemoryType.SCALAR], [0, 0]),    # 1: r1 = r0 * r0 (effective)
        Instruction(ScalarAddOp(), MemoryType.SCALAR, 9, 
                   [MemoryType.SCALAR, MemoryType.SCALAR], [1, 1]),    # 2: r9 = r1 + r1 (effective, output)
        Instruction(ScalarSubOp(), MemoryType.SCALAR, 2, 
                   [MemoryType.SCALAR, MemoryType.SCALAR], [0, 1]),     # 3: r2 = r0 - r1 (INTRON)
        Instruction(ScalarMulOp(), MemoryType.SCALAR, 3, 
                   [MemoryType.SCALAR, MemoryType.SCALAR], [2, 2]),    # 4: r3 = r2 * r2 (INTRON)
        Instruction(ScalarAddOp(), MemoryType.SCALAR, 4, 
                   [MemoryType.SCALAR, MemoryType.SCALAR], [3, 3]),     # 5: r4 = r3 + r3 (INTRON)
    ]
    prog2 = Program(instructions2.copy())
    print(f"Before: {len(prog2)} instructions")
    for i, instr in enumerate(prog2.instructions):
        print(f"  {i}: r{instr.dest_index} = ...")
    
    prog2.intron_removal([(MemoryType.SCALAR, 9)])
    print(f"After: {len(prog2)} instructions")
    for i, instr in enumerate(prog2.instructions):
        print(f"  {i}: r{instr.dest_index} = ...")
    assert len(prog2) == 3, f"Expected 3 instructions, got {len(prog2)}"
    assert prog2.instructions[0].dest_index == 0, "First instruction should write to r0"
    assert prog2.instructions[1].dest_index == 1, "Second instruction should write to r1"
    assert prog2.instructions[2].dest_index == 9, "Third instruction should write to r9"
    print("✅ Test 2 passed!")
    
    # Test 3: Overwritten registers - later instruction overwrites earlier
    print("\n--- Test 3: Overwritten registers ---")
    instructions3 = [
        Instruction(ScalarAddOp(), MemoryType.SCALAR, 0, 
                   [MemoryType.SCALAR, MemoryType.SCALAR], [-1, -2]),  # 0: r0 = obs[-1] + obs[-2] (INTRON - overwritten)
        Instruction(ScalarMulOp(), MemoryType.SCALAR, 0, 
                   [MemoryType.SCALAR, MemoryType.SCALAR], [-1, -2]),  # 1: r0 = obs[-1] * obs[-2] (effective)
        Instruction(ScalarAddOp(), MemoryType.SCALAR, 9, 
                   [MemoryType.SCALAR, MemoryType.SCALAR], [0, 0]),     # 2: r9 = r0 + r0 (effective, output)
    ]
    prog3 = Program(instructions3.copy())
    print(f"Before: {len(prog3)} instructions")
    for i, instr in enumerate(prog3.instructions):
        print(f"  {i}: r{instr.dest_index} = ...")
    
    prog3.intron_removal([(MemoryType.SCALAR, 9)])
    print(f"After: {len(prog3)} instructions")
    for i, instr in enumerate(prog3.instructions):
        print(f"  {i}: r{instr.dest_index} = ...")
    assert len(prog3) == 2, f"Expected 2 instructions, got {len(prog3)}"
    assert prog3.instructions[0].dest_index == 0, "First instruction should write to r0"
    assert prog3.instructions[1].dest_index == 9, "Second instruction should write to r9"
    print("✅ Test 3 passed!")
    
    # Test 4: Multiple output registers
    print("\n--- Test 4: Multiple output registers ---")
    instructions4 = [
        Instruction(ScalarAddOp(), MemoryType.SCALAR, 0, 
                   [MemoryType.SCALAR, MemoryType.SCALAR], [-1, -2]),    # 0: r0 = obs[-1] + obs[-2]
        Instruction(ScalarMulOp(), MemoryType.SCALAR, 1, 
                   [MemoryType.SCALAR, MemoryType.SCALAR], [0, 0]),     # 1: r1 = r0 * r0
        Instruction(ScalarAddOp(), MemoryType.SCALAR, 9, 
                   [MemoryType.SCALAR, MemoryType.SCALAR], [1, 1]),    # 2: r9 = r1 + r1 (output 1)
        Instruction(ScalarSubOp(), MemoryType.SCALAR, 8, 
                   [MemoryType.SCALAR, MemoryType.SCALAR], [0, 1]),    # 3: r8 = r0 - r1 (output 2)
        Instruction(ScalarMulOp(), MemoryType.SCALAR, 7, 
                   [MemoryType.SCALAR, MemoryType.SCALAR], [2, 2]),    # 4: r7 = r2 * r2 (INTRON - r2 never written)
    ]
    prog4 = Program(instructions4.copy())
    print(f"Before: {len(prog4)} instructions")
    for i, instr in enumerate(prog4.instructions):
        print(f"  {i}: r{instr.dest_index} = ...")
    
    prog4.intron_removal([(MemoryType.SCALAR, 9), (MemoryType.SCALAR, 8)])
    print(f"After: {len(prog4)} instructions")
    for i, instr in enumerate(prog4.instructions):
        print(f"  {i}: r{instr.dest_index} = ...")
    assert len(prog4) == 4, f"Expected 4 instructions, got {len(prog4)}"
    print("✅ Test 4 passed!")
    
    # Test 5: Empty program
    print("\n--- Test 5: Empty program ---")
    prog5 = Program([])
    print(f"Before: {len(prog5)} instructions")
    prog5.intron_removal([(MemoryType.SCALAR, 9)])
    print(f"After: {len(prog5)} instructions")
    assert len(prog5) == 0, f"Expected 0 instructions, got {len(prog5)}"
    print("✅ Test 5 passed!")
    
    # Test 6: Single instruction
    print("\n--- Test 6: Single instruction ---")
    instructions6 = [
        Instruction(ScalarAddOp(), MemoryType.SCALAR, 9, 
                   [MemoryType.SCALAR, MemoryType.SCALAR], [-1, -2]),  # 0: r9 = obs[-1] + obs[-2] (output)
    ]
    prog6 = Program(instructions6.copy())
    print(f"Before: {len(prog6)} instructions")
    prog6.intron_removal([(MemoryType.SCALAR, 9)])
    print(f"After: {len(prog6)} instructions")
    assert len(prog6) == 1, f"Expected 1 instruction, got {len(prog6)}"
    print("✅ Test 6 passed!")
    
    # Test 7: All instructions are introns (output register never written)
    print("\n--- Test 7: All instructions are introns ---")
    instructions7 = [
        Instruction(ScalarAddOp(), MemoryType.SCALAR, 0, 
                   [MemoryType.SCALAR, MemoryType.SCALAR], [-1, -2]),  # 0: r0 = obs[-1] + obs[-2]
        Instruction(ScalarMulOp(), MemoryType.SCALAR, 1, 
                   [MemoryType.SCALAR, MemoryType.SCALAR], [0, 0]),   # 1: r1 = r0 * r0
    ]
    prog7 = Program(instructions7.copy())
    print(f"Before: {len(prog7)} instructions")
    prog7.intron_removal([(MemoryType.SCALAR, 9)])  # r9 is never written
    print(f"After: {len(prog7)} instructions")
    assert len(prog7) == 0, f"Expected 0 instructions, got {len(prog7)}"
    print("✅ Test 7 passed!")
    
    # Test 8: Complex dependency chain
    print("\n--- Test 8: Complex dependency chain ---")
    instructions8 = [
        Instruction(ScalarAddOp(), MemoryType.SCALAR, 0, 
                   [MemoryType.SCALAR, MemoryType.SCALAR], [-1, -2]),  # 0: r0 = obs[-1] + obs[-2]
        Instruction(ScalarMulOp(), MemoryType.SCALAR, 1, 
                   [MemoryType.SCALAR, MemoryType.SCALAR], [0, 0]),     # 1: r1 = r0 * r0
        Instruction(ScalarSubOp(), MemoryType.SCALAR, 2, 
                   [MemoryType.SCALAR, MemoryType.SCALAR], [1, 0]),    # 2: r2 = r1 - r0
        Instruction(ScalarDivProtectedOp(), MemoryType.SCALAR, 3, 
                   [MemoryType.SCALAR, MemoryType.SCALAR], [2, 1]),    # 3: r3 = r2 / r1
        Instruction(ScalarAddOp(), MemoryType.SCALAR, 9, 
                   [MemoryType.SCALAR, MemoryType.SCALAR], [3, 3]),    # 4: r9 = r3 + r3 (output)
        Instruction(ScalarMulOp(), MemoryType.SCALAR, 4, 
                   [MemoryType.SCALAR, MemoryType.SCALAR], [5, 5]),    # 5: r4 = r5 * r5 (INTRON - r5 never written)
    ]
    prog8 = Program(instructions8.copy())
    print(f"Before: {len(prog8)} instructions")
    for i, instr in enumerate(prog8.instructions):
        print(f"  {i}: r{instr.dest_index} = ...")
    
    prog8.intron_removal([(MemoryType.SCALAR, 9)])
    print(f"After: {len(prog8)} instructions")
    for i, instr in enumerate(prog8.instructions):
        print(f"  {i}: r{instr.dest_index} = ...")
    assert len(prog8) == 5, f"Expected 5 instructions, got {len(prog8)}"
    assert all(instr.dest_index != 4 for instr in prog8.instructions), "r4 instruction should be removed"
    print("✅ Test 8 passed!")
    
    # Test 9: Test that remove_introns_create_copy works correctly
    print("\n--- Test 9: remove_introns_create_copy (non-destructive) ---")
    instructions9 = [
        Instruction(ScalarAddOp(), MemoryType.SCALAR, 0, 
                   [MemoryType.SCALAR, MemoryType.SCALAR], [-1, -2]),
        Instruction(ScalarMulOp(), MemoryType.SCALAR, 1, 
                   [MemoryType.SCALAR, MemoryType.SCALAR], [0, 0]),
        Instruction(ScalarAddOp(), MemoryType.SCALAR, 9, 
                   [MemoryType.SCALAR, MemoryType.SCALAR], [1, 1]),
        Instruction(ScalarSubOp(), MemoryType.SCALAR, 2, 
                   [MemoryType.SCALAR, MemoryType.SCALAR], [0, 1]),  # INTRON
    ]
    original = Program(instructions9.copy())
    print(f"Original: {len(original)} instructions")
    
    pruned = original.remove_introns_create_copy([(MemoryType.SCALAR, 9)])
    print(f"Pruned copy: {len(pruned)} instructions")
    print(f"Original (should be unchanged): {len(original)} instructions")
    
    assert len(original) == 4, "Original should remain unchanged"
    assert len(pruned) == 3, "Pruned copy should have 3 instructions"
    print("✅ Test 9 passed!")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)