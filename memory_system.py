"""
Memory System for Linear Genetic Programming (LGP).

This module provides a typed memory bank system for managing registers in LGP programs.
The memory bank is divided into two regions:
    - Observation registers: Read-only memory for environment inputs
    - Working registers: Read-write memory for program computation

All data is stored as numpy arrays with dtype=np.float32 for efficiency.
"""

from dataclasses import dataclass
from enum import Enum
import numpy as np
from typing import Tuple

class MemoryType(Enum):
    """
    Enumeration of memory register types.
    
    Attributes:
        SCALAR: Single floating-point value
        VECTOR: 1D array of floating-point values
        MATRIX: 2D array of floating-point values
    """
    SCALAR = "scalar"
    VECTOR = "vector"
    MATRIX = "matrix"


@dataclass
class MemoryConfig:
    n_scalar: int
    n_vector: int
    n_matrix: int
    n_obs_scalar: int
    n_obs_vector: int
    n_obs_matrix: int
    vector_size: int
    matrix_shape: Tuple[int, int]
    init_scalar_range: Tuple[float, float] = (-2.0, 2.0)
    init_vector_range: Tuple[float, float] = (-1.0, 1.0)
    init_matrix_range: Tuple[float, float] = (-0.5, 0.5)

class MemoryBank:
    """
    Manages typed memory registers for LGP programs.
    
    The MemoryBank provides two separate memory regions:
    
    1. **Observation Registers** (read-only):
       - Stores environment inputs and observations
       - Can only be read, never written by the program
       - Used for sensor data, state information, etc.
    
    2. **Working Registers** (read-write):
       - Stores intermediate computation results
       - Can be read and written by the program
       - Used for program execution and state management
    
    All registers support automatic wrapping: if an index exceeds the register
    count, it wraps around using modulo arithmetic. This prevents index errors
    and allows for flexible program behavior.
    
    Attributes:
        n_scalar (int): Number of scalar working registers
        n_vector (int): Number of vector working registers
        n_matrix (int): Number of matrix working registers
        n_obs_scalar (int): Number of scalar observation registers
        n_obs_vector (int): Number of vector observation registers
        n_obs_matrix (int): Number of matrix observation registers
        vector_size (int): Size of each vector register
        matrix_shape (tuple): Shape (height, width) of each matrix register
        
    Example:
        >>> memory = MemoryBank(
        ...     n_scalar=8, n_vector=4, n_matrix=2,
        ...     n_obs_scalar=2, n_obs_vector=1, n_obs_matrix=1,
        ...     vector_size=10, matrix_shape=(84, 84)
        ... )
        >>> memory.write_scalar(0, 42.0)
        >>> value = memory.read_scalar(0)
        >>> print(value)
        42.0
    """
    def __init__(self,
                 n_scalar: int,
                 n_vector: int,
                 n_matrix: int,
                 n_obs_scalar: int,
                 n_obs_vector: int,
                 n_obs_matrix: int,
                 vector_size: int,
                 matrix_shape: tuple,
                 # Random initialization parameters
                 init_scalar_range: Tuple[float, float] = (-2.0, 2.0),
                 init_vector_range: Tuple[float, float] = (-1.0, 1.0),
                 init_matrix_range: Tuple[float, float] = (-0.5, 0.5),
                 rng: np.random.Generator = None  # Optional RNG for reproducibility
                 ):
        """
        Initialize a MemoryBank with specified register counts and dimensions.
        
        All registers are initialized to 0.1 (not zero) to avoid issues with
        uninitialized memory in genetic programs.
        
        Args:
            n_scalar: Number of scalar working registers to allocate
            n_vector: Number of vector working registers to allocate
            n_matrix: Number of matrix working registers to allocate
            n_obs_scalar: Number of scalar observation registers (read-only)
            n_obs_vector: Number of vector observation registers (read-only)
            n_obs_matrix: Number of matrix observation registers (read-only)
            vector_size: Size of each vector register (all vectors same size)
            matrix_shape: Tuple (height, width) for matrix register dimensions
            rng: Optional numpy random Generator for reproducible initialization
            
        Raises:
            ValueError: If any count is negative or dimensions are invalid
            
        Note:
            All data is stored as np.float32 for memory efficiency and
            consistency with typical LGP implementations.
        """
        self.n_scalar = n_scalar
        self.n_vector = n_vector
        self.n_matrix = n_matrix
        self.vector_size = vector_size
        self.matrix_shape = matrix_shape
        self.n_obs_scalar = n_obs_scalar
        self.n_obs_vector = n_obs_vector
        self.n_obs_matrix = n_obs_matrix

        # OBSERVATION READ ONLY
        self.obs_scalars = np.full(n_obs_scalar, 0.1, dtype=np.float32)
        self.obs_vectors = np.full((n_obs_vector, vector_size), 0.1, dtype=np.float32)
        self.obs_matrices = np.full((n_obs_matrix, *matrix_shape), 0.1, dtype=np.float32)
        
        # self.scalars = np.full(n_scalar, 0.1, dtype=np.float32)
        # self.vectors = np.full((n_vector, vector_size), 0.1, dtype=np.float32)
        # self.matrices = np.full((n_matrix, *matrix_shape), 0.1, dtype=np.float32)

        # Use the passed RNG or create a default one for reproducibility
        if rng is None:
            rng = np.random.default_rng()

        # WORKING MEMORY - initialized with RANDOM values (evolvable constants)
        self.scalars = rng.uniform(
            init_scalar_range[0], init_scalar_range[1], n_scalar
        ).astype(np.float32)
        
        self.vectors = rng.uniform(
            init_vector_range[0], init_vector_range[1], (n_vector, vector_size)
        ).astype(np.float32)
        
        self.matrices = rng.uniform(
            init_matrix_range[0], init_matrix_range[1], (n_matrix, *matrix_shape)
        ).astype(np.float32)
     
    # ========== Observation Register Read Methods (Read-Only) ==========
    
    def read_obs_scalar(self, index: int) -> float:
        """
        Read a scalar value from observation registers.
        
        Args:
            index: Register index (wraps around if out of bounds)
            
        Returns:
            float: The scalar value at the specified index
            
        Example:
            >>> value = memory.read_obs_scalar(0)
            >>> # Index 10 wraps to (10 % n_obs_scalar)
        """
        index = index % self.n_obs_scalar
        return self.obs_scalars[index]
    
    def read_obs_vector(self, index: int) -> np.ndarray:
        """
        Read a vector from observation registers.
        
        Args:
            index: Register index (wraps around if out of bounds)
            
        Returns:
            np.ndarray: A 1D array of shape (vector_size,) containing the vector
                       data. Returns a view, not a copy.
                       
        Example:
            >>> vec = memory.read_obs_vector(0)
            >>> print(vec.shape)  # (10,) if vector_size=10
        """
        index = index % self.n_obs_vector
        return self.obs_vectors[index]
    
    def read_obs_matrix(self, index: int) -> np.ndarray:
        """
        Read a matrix from observation registers.
        
        Args:
            index: Register index (wraps around if out of bounds)
            
        Returns:
            np.ndarray: A 2D array of shape matrix_shape containing the matrix
                       data. Returns a view, not a copy.
                       
        Example:
            >>> mat = memory.read_obs_matrix(0)
            >>> print(mat.shape)  # (84, 84) if matrix_shape=(84, 84)
        """
        index = index % self.n_obs_matrix
        return self.obs_matrices[index]
    
    # ========== Working Register Read Methods ==========
    
    def read_scalar(self, index: int) -> float:
        """
        Read a scalar value from working registers.
        
        Args:
            index: Register index (wraps around if out of bounds)
            
        Returns:
            float: The scalar value at the specified index
        """
        index = index % self.n_scalar
        return self.scalars[index]
    
    def read_vector(self, index: int) -> np.ndarray:
        """
        Read a vector from working registers.
        
        Args:
            index: Register index (wraps around if out of bounds)
            
        Returns:
            np.ndarray: A 1D array of shape (vector_size,) containing the vector
                       data. Returns a view, not a copy.
        """
        index = index % self.n_vector
        return self.vectors[index]
    
    def read_matrix(self, index: int) -> np.ndarray:
        """
        Read a matrix from working registers.
        
        Args:
            index: Register index (wraps around if out of bounds)
            
        Returns:
            np.ndarray: A 2D array of shape matrix_shape containing the matrix
                       data. Returns a view, not a copy.
        """
        index = index % self.n_matrix
        return self.matrices[index]

    # ========== Working Register Write Methods ==========
    
    def write_scalar(self, index: int, value: float) -> None:
        """
        Write a scalar value to working registers.
        
        Args:
            index: Register index (wraps around if out of bounds)
            value: Scalar value to write
            
        Example:
            >>> memory.write_scalar(0, 42.0)
            >>> memory.write_scalar(100, 3.14)  # Wraps if n_scalar < 100
        """
        index = index % self.n_scalar
        # Clamp to reasonable range for float32 to prevent overflow
        if not np.isfinite(value):
            value = 0.0
        else:
            # float32 range: approximately [-3.4e38, 3.4e38]
            # Use a safer range to prevent overflow
            value = np.clip(value, -1e10, 1e10)
        self.scalars[index] = float(value)
    
    def write_vector(self, index: int, value: np.ndarray) -> None:
        """
        Write a vector to working registers.
        
        Automatically handles size mismatches:
        - If value is smaller: pads with zeros
        - If value is larger: truncates to vector_size
        - If value matches: copies directly
        
        Uses np.copyto to ensure data is copied into the existing buffer
        rather than creating a reference.
        
        Args:
            index: Register index (wraps around if out of bounds)
            value: Vector to write. Can be 1D array, list, or scalar (converted to 1D)
            
        Example:
            >>> memory.write_vector(0, [1, 2, 3])  # Pads if vector_size > 3
            >>> memory.write_vector(1, np.arange(20))  # Truncates if vector_size < 20
        """
        index = index % self.n_vector
        value = np.atleast_1d(value).astype(np.float32)
        
        # Handle size mismatch
        if value.shape[0] == self.vector_size:
            np.copyto(self.vectors[index], value)
        elif value.shape[0] < self.vector_size:
            # Pad with zeros
            self.vectors[index] = 0
            self.vectors[index, :value.shape[0]] = value
        else:
            # Truncate
            np.copyto(self.vectors[index], value[:self.vector_size])
    
    def write_matrix(self, index: int, value: np.ndarray) -> None:
        """
        Write a matrix to working registers.
        
        Automatically handles shape mismatches using center alignment:
        - If value is smaller: pads with zeros (centered)
        - If value is larger: crops from center
        - If value matches: copies directly
        
        Uses center alignment to preserve spatial relationships in image-like
        data. For example, a 3x3 matrix into a 5x5 target will be centered
        with padding on all sides.
        
        Uses np.copyto when shapes match to ensure data is copied into the
        existing buffer rather than creating a reference.
        
        Args:
            index: Register index (wraps around if out of bounds)
            value: Matrix to write. Can be 2D array or will be converted to 2D
            
        Example:
            >>> # Write 3x3 matrix to 5x5 register (centered with padding)
            >>> small = np.ones((3, 3))
            >>> memory.write_matrix(0, small)
            >>> # Result: zeros with 3x3 ones in center
            
            >>> # Write 10x10 matrix to 5x5 register (center crop)
            >>> large = np.arange(100).reshape(10, 10)
            >>> memory.write_matrix(0, large)
            >>> # Result: center 5x5 region of original matrix
        """
        index = index % self.n_matrix
        
        value = np.atleast_2d(value).astype(np.float32)
        
        # Handle shape mismatch - center crop or pad
        h, w = value.shape[:2]
        target_h, target_w = self.matrix_shape
        
        if (h, w) == (target_h, target_w):
            np.copyto(self.matrices[index], value)
        else:
            # Reset to zero
            self.matrices[index].fill(0)
            
            # Calculate copy region
            copy_h = min(h, target_h)
            copy_w = min(w, target_w)
            
            # Center alignment
            src_h_start = (h - copy_h) // 2
            src_w_start = (w - copy_w) // 2
            dst_h_start = (target_h - copy_h) // 2
            dst_w_start = (target_w - copy_w) // 2
            
            self.matrices[index][
                dst_h_start:dst_h_start + copy_h,
                dst_w_start:dst_w_start + copy_w
            ] = value[
                src_h_start:src_h_start + copy_h,
                src_w_start:src_w_start + copy_w
            ]

    def load_observation(self, obs: dict) -> None:
        """
        Load observation data into observation registers (read-only memory).
        
        This method is used to update the observation registers with new
        environment data. The observation registers are read-only from the
        program's perspective but can be updated externally via this method.
        
        Args:
            obs: Dictionary containing observation data. Keys can be:
                - 'scalar': List or array of scalar values
                - 'vector': List or array of vectors (will be converted to 2D)
                - 'matrix': List or array of matrices
                
                Only the keys present in the dictionary will be updated.
                If fewer values are provided than registers, only the first
                N registers are updated.
                
        Example:
            >>> memory.load_observation({
            ...     'scalar': [1.5, 2.5],
            ...     'vector': [[1, 2, 3], [4, 5, 6]],
            ...     'matrix': [np.random.rand(84, 84)]
            ... })
        """
        if 'scalar' in obs:
            obs_data = np.atleast_1d(obs['scalar'])
            self.obs_scalars[:len(obs_data)] = obs_data
        
        if 'vector' in obs:
            obs_data = np.atleast_2d(obs['vector'])
            self.obs_vectors[:len(obs_data)] = obs_data
        
        if 'matrix' in obs:
            obs_data = np.array(obs['matrix'])
            self.obs_matrices[:len(obs_data)] = obs_data
        
    # ========== State Management Methods ==========
    
    def reset(self) -> None:
        """
        Reset all working registers to initial value (0.1).
        
        Observation registers are left unchanged. This is useful for
        resetting program state between episodes or iterations while
        preserving observation data.
        
        Note:
            Registers are reset to 0.1 (not zero) to avoid issues with
            uninitialized memory in genetic programs.
            
        Example:
            >>> memory.write_scalar(0, 42.0)
            >>> memory.reset()
            >>> print(memory.read_scalar(0))  # 0.1
            >>> print(memory.read_obs_scalar(0))  # Unchanged
        """
        self.scalars.fill(0.1)
        self.vectors.fill(0.1)
        self.matrices.fill(0.1)
        
    def reset_all(self) -> None:
        """
        Reset both working and observation registers to initial value (0.1).
        
        This performs a complete reset of all memory, including observations.
        Use this when starting a completely new episode or simulation.
        
        Note:
            Registers are reset to 0.1 (not zero) to avoid issues with
            uninitialized memory in genetic programs.
            
        Example:
            >>> memory.reset_all()
            >>> # All registers (working and observation) are now 0.1
        """
        self.reset()
        self.obs_scalars.fill(0.1)
        self.obs_vectors.fill(0.1)
        self.obs_matrices.fill(0.1)
    def copy(self) -> 'MemoryBank':
        """
        Create a deep copy of this memory bank.
        
        Returns a new MemoryBank instance with identical configuration and
        a complete copy of all register data. The new instance is completely
        independent - modifications to one will not affect the other.
        
        Returns:
            MemoryBank: A new MemoryBank instance with copied state
            
        Example:
            >>> memory2 = memory.copy()
            >>> memory2.write_scalar(0, 99.0)
            >>> print(memory.read_scalar(0))  # Original unchanged
            >>> print(memory2.read_scalar(0))  # 99.0
        """
        new_bank = MemoryBank(
            n_scalar=self.n_scalar,
            n_vector=self.n_vector,
            n_matrix=self.n_matrix,
            n_obs_scalar=self.n_obs_scalar,
            n_obs_vector=self.n_obs_vector,
            n_obs_matrix=self.n_obs_matrix,
            vector_size=self.vector_size,
            matrix_shape=self.matrix_shape
        )
        
        # Copy observation state
        np.copyto(new_bank.obs_scalars, self.obs_scalars)
        np.copyto(new_bank.obs_vectors, self.obs_vectors)
        np.copyto(new_bank.obs_matrices, self.obs_matrices)
        
        # Copy working state
        np.copyto(new_bank.scalars, self.scalars)
        np.copyto(new_bank.vectors, self.vectors)
        np.copyto(new_bank.matrices, self.matrices)
        
        return new_bank
    
    def __repr__(self) -> str:
        """
        Return a string representation of the MemoryBank.
        
        Returns:
            str: String showing register counts in format:
                 "MemoryBank(obs=[Xs, Yv, Zm], registers=[As, Bv, Cm])"
                 where X/Y/Z are observation counts and A/B/C are working counts
        """
        return (f"MemoryBank(obs=[{self.n_obs_scalar}s, {self.n_obs_vector}v, {self.n_obs_matrix}m], "
                f"registers=[{self.n_scalar}s, {self.n_vector}v, {self.n_matrix}m])")
if __name__ == "__main__":
    # Quick test
    print("Testing simplified MemoryBank...")
    
    memory = MemoryBank(
        n_scalar=8,
        n_vector=4,
        n_matrix=2,
        n_obs_scalar=2,
        n_obs_vector=1,
        n_obs_matrix=1,
        vector_size=10,
        matrix_shape=(84, 84)
    )
    
    print(f"\n{memory}")
    
    # Load observations
    memory.load_observation({
        'scalar': [1.5, 2.5],
        'matrix': [np.random.rand(84, 84)]
    })
    
    print(f"\nObservation scalars: {memory.read_obs_scalar(0)}, {memory.read_obs_scalar(1)}")
    print(f"Observation matrix mean: {memory.read_obs_matrix(0).mean():.4f}")
    
    # Write to working registers
    memory.write_scalar(0, 42.0)
    print(f"\nWorking scalar[0]: {memory.read_scalar(0)}")
    print(memory)
    
    # Reset
    memory.reset()
    print(f"After reset, working scalar[0]: {memory.read_scalar(0)}")
    print(f"After reset, obs scalar[0]: {memory.read_obs_scalar(0)} (unchanged)")
    
    print("\n Simplified MemoryBank works!")
