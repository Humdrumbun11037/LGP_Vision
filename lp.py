#!/usr/bin/env python3
"""
Bellman-Ford Algorithm for Q2 Constraint Graph
Solves the system of difference constraints
"""

def bellman_ford(vertices, edges, source):
    """
    Bellman-Ford algorithm to find shortest paths from source
    
    Args:
        vertices: list of vertex names
        edges: list of tuples (u, v, weight)
        source: source vertex
    
    Returns:
        distances: dict of shortest distances from source
        predecessors: dict of predecessors for path reconstruction
        has_negative_cycle: boolean indicating if negative cycle exists
    """
    # Initialize distances
    distances = {v: float('inf') for v in vertices}
    distances[source] = 0
    predecessors = {v: None for v in vertices}
    
    n = len(vertices)
    
    print("Initial distances:")
    print(distances)
    print("\n" + "="*60)
    
    # Relax edges |V|-1 times
    for iteration in range(1, n):
        print(f"\n--- ITERATION {iteration} ---")
        updated = False
        edges_relaxed = []
        
        for u, v, weight in edges:
            if distances[u] != float('inf'):
                new_distance = distances[u] + weight
                if new_distance < distances[v]:
                    distances[v] = new_distance
                    predecessors[v] = u
                    updated = True
                    edges_relaxed.append((u, v, weight))
                    print(f"  Relaxed edge ({u} -> {v}, {weight}): d[{v}] = {distances[u]} + {weight} = {new_distance}")
        
        if not updated:
            print(f"  No edges relaxed - algorithm converged!")
            print(f"\nFinal distances after iteration {iteration}:")
            for v in sorted(vertices):
                if v != source:
                    print(f"  d[{v}] = {distances[v]}")
            return distances, predecessors, False
        
        print(f"\n  Distances after iteration {iteration}:")
        for v in sorted(vertices):
            if v != source:
                print(f"    d[{v}] = {distances[v]}")
    
    # Check for negative cycles
    print(f"\n--- CHECKING FOR NEGATIVE CYCLES (Iteration {n}) ---")
    for u, v, weight in edges:
        if distances[u] != float('inf'):
            if distances[u] + weight < distances[v]:
                print(f"  Edge ({u} -> {v}, {weight}) can still be relaxed!")
                print(f"  NEGATIVE CYCLE DETECTED!")
                return distances, predecessors, True
    
    print("  No negative cycle detected.")
    print(f"\nFinal distances:")
    for v in sorted(vertices):
        if v != source:
            print(f"  d[{v}] = {distances[v]}")
    
    return distances, predecessors, False


def verify_constraints(solution, constraints):
    """
    Verify that the solution satisfies all difference constraints
    
    Args:
        solution: dict of variable values
        constraints: list of tuples (xi, xj, bound) representing xj - xi <= bound
    """
    print("\n" + "="*60)
    print("CONSTRAINT VERIFICATION")
    print("="*60)
    
    all_satisfied = True
    for i, (xi, xj, bound) in enumerate(constraints, 1):
        diff = solution[xj] - solution[xi]
        satisfied = diff <= bound
        status = "✓" if satisfied else "✗"
        print(f"{status} Constraint {i}: x_{xj} - x_{xi} = {solution[xj]} - {solution[xi]} = {diff} <= {bound}")
        if not satisfied:
            all_satisfied = False
    
    print("\n" + "="*60)
    if all_satisfied:
        print("✓✓✓ ALL CONSTRAINTS SATISFIED! ✓✓✓")
    else:
        print("✗✗✗ SOME CONSTRAINTS VIOLATED! ✗✗✗")
    print("="*60)


def main():
    print("="*60)
    print("BELLMAN-FORD FOR Q2 CONSTRAINT GRAPH")
    print("="*60)
    
    # Define vertices
    vertices = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    
    # Define edges: (source, destination, weight)
    edges = [
        # Source edges (weight 0)
        ('x0', 'x1', 0),
        ('x0', 'x2', 0),
        ('x0', 'x3', 0),
        ('x0', 'x4', 0),
        ('x0', 'x5', 0),
        ('x0', 'x6', 0),
        # Constraint edges
        ('x1', 'x2', 7),   # x2 - x1 <= 7
        ('x1', 'x3', 2),   # x3 - x1 <= 2
        ('x2', 'x4', 3),   # x4 - x2 <= 3
        ('x3', 'x4', 6),   # x4 - x3 <= 6
        ('x3', 'x5', 1),   # x5 - x3 <= 1
        ('x4', 'x6', 4),   # x6 - x4 <= 4
        ('x6', 'x5', -2),  # x5 - x6 <= -2
        ('x5', 'x2', 5),   # x2 - x5 <= 5
        ('x6', 'x3', 3),   # x3 - x6 <= 3
    ]
    
    print(f"\nNumber of vertices: {len(vertices)}")
    print(f"Number of edges: {len(edges)}")
    print(f"Source node: x0")
    
    # Run Bellman-Ford
    distances, predecessors, has_negative_cycle = bellman_ford(vertices, edges, 'x0')
    
    print("\n" + "="*60)
    if has_negative_cycle:
        print("RESULT: NEGATIVE CYCLE EXISTS - NO FEASIBLE SOLUTION")
    else:
        print("RESULT: NO NEGATIVE CYCLE - FEASIBLE SOLUTION EXISTS")
        print("="*60)
        print("\nFEASIBLE SOLUTION:")
        print("-"*60)
        for v in sorted(vertices):
            if v != 'x0':
                print(f"  {v} = {distances[v]}")
        
        # Verify the original constraints
        # Format: (xi, xj, bound) means xj - xi <= bound
        original_constraints = [
            (1, 2, 7),   # x2 - x1 <= 7
            (1, 3, 2),   # x3 - x1 <= 2
            (2, 4, 3),   # x4 - x2 <= 3
            (3, 4, 6),   # x4 - x3 <= 6
            (3, 5, 1),   # x5 - x3 <= 1
            (4, 6, 4),   # x6 - x4 <= 4
            (6, 5, -2),  # x5 - x6 <= -2
            (5, 2, 5),   # x2 - x5 <= 5
            (6, 3, 3),   # x3 - x6 <= 3
        ]
        
        # Convert to use variable names
        solution = {f'x{i}': distances[f'x{i}'] for i in range(1, 7)}
        constraints_with_names = [(f'x{xi}', f'x{xj}', bound) for xi, xj, bound in original_constraints]
        
        verify_constraints(solution, constraints_with_names)


if __name__ == "__main__":
    main()