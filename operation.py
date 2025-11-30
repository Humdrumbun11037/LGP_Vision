import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Any
from memory_system import MemoryType

# CV operations dependencies

import cv2


try:
    from scipy import ndimage
except ImportError:
    ndimage = None

SAFE_FLOAT_RANGE = 1e6


def _clip_array(arr: np.ndarray, limit: float = SAFE_FLOAT_RANGE) -> np.ndarray:
    """Clamp array values to a safe float range to avoid overflow."""
    return np.clip(arr, -limit, limit)

class Operation(ABC):
    """Abstract class for all operations 
    Each operations know its input and output types and how to execute it"""
    @abstractmethod
    def input_types(self) -> List[MemoryType]:
        """Return the input types of the operation"""
        pass

    @abstractmethod
    def output_type(self) -> MemoryType:
        """Return the output type of the operation"""
        pass

    @abstractmethod
    def execute(self, inputs: List[Any]) -> Any:
        """Execute the operation"""
        pass
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the operation"""
        pass
    @property
    @abstractmethod
    def differentiable(self) -> bool:
        """Return if the operation is differentiable"""
        pass
    def __repr__(self) -> str:
        """Return the string representation of the operation"""
        input_str = ", ".join([t.value for t in self.input_types()])
        return f"{self.name}({input_str}) -> {self.output_type().value}"
    #============================= SCALAR OPERATIONS =============================
class ScalarAddOp(Operation):
    """Add two scalar values"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR, MemoryType.SCALAR]
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    def execute(self, a: float, b: float) -> float:   
        return a + b
    @property
    def name(self) -> str:
        return "scalar_add"
    @property
    def differentiable(self) -> bool:
        return True

class ScalarSubOp(Operation):
    """Subtract two scalar values"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR, MemoryType.SCALAR]
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    def execute(self, a: float, b: float) -> float:   
        return a - b
    @property
    def name(self) -> str:
        return "scalar_sub"
    @property
    def differentiable(self) -> bool:
        return True
    
class ScalarMulOp(Operation):
    """Multiply two scalars: a * b"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR, MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, a: float, b: float) -> float:
        return a * b
    
    @property
    def name(self) -> str:
        return "scalar_mul"
    
    @property
    def differentiable(self) -> bool:
        return True


class ScalarDivProtectedOp(Operation):
    """Protected division: a / b (returns 1.0 if b == 0)"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR, MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, a: float, b: float) -> float:
        return a / b if abs(b) > 1e-8 else 1.0
    
    @property
    def name(self) -> str:
        return "scalar_div_protected"
    
    @property
    def differentiable(self) -> bool:
        return False  # Has discontinuity at b=0
class ScalarMaxOp(Operation):
    """Maximum of two scalars: max(a, b)"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR, MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, a: float, b: float) -> float:
        return max(a, b)
    
    @property
    def name(self) -> str:
        return "scalar_max"
    
    @property
    def differentiable(self) -> bool:
        return False  # Not differentiable at a=b


class ScalarMinOp(Operation):
    """Minimum of two scalars: min(a, b)"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR, MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, a: float, b: float) -> float:
        return min(a, b)
    
    @property
    def name(self) -> str:
        return "scalar_min"
    
    @property
    def differentiable(self) -> bool:
        return False  # Not differentiable at a=b


class ScalarAbsOp(Operation):
    """Absolute value: |a|"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, a: float) -> float:
        return abs(a)
    
    @property
    def name(self) -> str:
        return "scalar_abs"
    
    @property
    def differentiable(self) -> bool:
        return False  # Not differentiable at a=0


class ScalarNegOp(Operation):
    """Negation: -a"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, a: float) -> float:
        return -a
    
    @property
    def name(self) -> str:
        return "scalar_neg"
    
    @property
    def differentiable(self) -> bool:
        return True
# ==================== VECTOR OPERATIONS ====================

class VectorAddOp(Operation):
    """Element-wise vector addition: a + b"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.VECTOR, MemoryType.VECTOR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.VECTOR
    
    def execute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b
    
    @property
    def name(self) -> str:
        return "vector_add"
    
    @property
    def differentiable(self) -> bool:
        return True


class VectorSubOp(Operation):
    """Element-wise vector subtraction: a - b"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.VECTOR, MemoryType.VECTOR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.VECTOR
    
    def execute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a - b
    
    @property
    def name(self) -> str:
        return "vector_sub"
    
    @property
    def differentiable(self) -> bool:
        return True


class VectorMulOp(Operation):
    """Element-wise vector multiplication: a * b"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.VECTOR, MemoryType.VECTOR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.VECTOR
    
    def execute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a * b
    
    @property
    def name(self) -> str:
        return "vector_mul"
    
    @property
    def differentiable(self) -> bool:
        return True


class VectorDotProductOp(Operation):
    """Dot product of two vectors: a · b → scalar"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.VECTOR, MemoryType.VECTOR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, a: np.ndarray, b: np.ndarray) -> float:
        a = _clip_array(a)
        b = _clip_array(b)
        result = np.dot(a, b)
        if not np.isfinite(result):
            result = 0.0
        return float(result)
    
    @property
    def name(self) -> str:
        return "vector_dot"
    
    @property
    def differentiable(self) -> bool:
        return True


class VectorMeanOp(Operation):
    """Mean of vector elements: mean(v) → scalar"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.VECTOR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, v: np.ndarray) -> float:
        v = _clip_array(v)
        result = np.mean(v)
        if not np.isfinite(result):
            result = 0.0
        return float(result)
    
    @property
    def name(self) -> str:
        return "vector_mean"
    
    @property
    def differentiable(self) -> bool:
        return True


class VectorMaxOp(Operation):
    """Maximum of vector elements: max(v) → scalar"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.VECTOR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, v: np.ndarray) -> float:
        return float(np.max(v))
    
    @property
    def name(self) -> str:
        return "vector_max"
    
    @property
    def differentiable(self) -> bool:
        return False  # Not differentiable at equality points


class VectorMinOp(Operation):
    """Minimum of vector elements: min(v) → scalar"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.VECTOR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, v: np.ndarray) -> float:
        return float(np.min(v))
    
    @property
    def name(self) -> str:
        return "vector_min"
    
    @property
    def differentiable(self) -> bool:
        return False  # Not differentiable at equality points


class VectorSumOp(Operation):
    """Sum of vector elements: sum(v) → scalar"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.VECTOR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, v: np.ndarray) -> float:
        v = _clip_array(v)
        result = np.sum(v)
        if not np.isfinite(result):
            result = 0.0
        return float(result)
    
    @property
    def name(self) -> str:
        return "vector_sum"
    
    @property
    def differentiable(self) -> bool:
        return True


class VectorNormOp(Operation):
    """L2 norm of vector: ||v|| → scalar"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.VECTOR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, v: np.ndarray) -> float:
        return float(np.linalg.norm(v))
    
    @property
    def name(self) -> str:
        return "vector_norm"
    
    @property
    def differentiable(self) -> bool:
        return True  # Differentiable everywhere except at v=0 (but commonly used in practice)
# ==================== MATRIX OPERATIONS ====================

class MatrixAddOp(Operation):
    """Element-wise matrix addition: A + B"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX, MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b
    
    @property
    def name(self) -> str:
        return "matrix_add"
    
    @property
    def differentiable(self) -> bool:
        return True


class MatrixSubOp(Operation):
    """Element-wise matrix subtraction: A - B"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX, MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a - b
    
    @property
    def name(self) -> str:
        return "matrix_sub"
    
    @property
    def differentiable(self) -> bool:
        return True


class MatrixMulOp(Operation):
    """Matrix multiplication: A @ B"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX, MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = _clip_array(a)
        b = _clip_array(b)
        result = np.matmul(a, b)
        result = _clip_array(result)
        # Replace any NaN/inf with 0
        result = np.nan_to_num(result, nan=0.0, posinf=SAFE_FLOAT_RANGE, neginf=-SAFE_FLOAT_RANGE)
        return result
    
    @property
    def name(self) -> str:
        return "matrix_mul"
    
    @property
    def differentiable(self) -> bool:
        return True


class MatrixMeanOp(Operation):
    """Global mean of matrix: mean(M) → scalar"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, m: np.ndarray) -> float:
        m = _clip_array(m)
        result = np.mean(m)
        if not np.isfinite(result):
            result = 0.0
        return float(result)
    
    @property
    def name(self) -> str:
        return "matrix_mean"
    
    @property
    def differentiable(self) -> bool:
        return True


class MatrixMaxOp(Operation):
    """Global maximum of matrix: max(M) → scalar"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, m: np.ndarray) -> float:
        return float(np.max(m))
    
    @property
    def name(self) -> str:
        return "matrix_max"
    
    @property
    def differentiable(self) -> bool:
        return False  # Not differentiable at equality points


class MatrixMinOp(Operation):
    """Global minimum of matrix: min(M) → scalar"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, m: np.ndarray) -> float:
        return float(np.min(m))
    
    @property
    def name(self) -> str:
        return "matrix_min"
    
    @property
    def differentiable(self) -> bool:
        return False  # Not differentiable at equality points


class MatrixRowMeanOp(Operation):
    """Mean across rows (axis=0): M → vector"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.VECTOR
    
    def execute(self, m: np.ndarray) -> np.ndarray:
        return np.mean(m, axis=0)
    
    @property
    def name(self) -> str:
        return "matrix_row_mean"
    
    @property
    def differentiable(self) -> bool:
        return True


class MatrixColMeanOp(Operation):
    """Mean across columns (axis=1): M → vector"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.VECTOR
    
    def execute(self, m: np.ndarray) -> np.ndarray:
        return np.mean(m, axis=1)
    
    @property
    def name(self) -> str:
        return "matrix_col_mean"
    
    @property
    def differentiable(self) -> bool:
        return True


class MatrixRowMaxOp(Operation):
    """Max across rows (axis=0): M → vector"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.VECTOR
    
    def execute(self, m: np.ndarray) -> np.ndarray:
        return np.max(m, axis=0)
    
    @property
    def name(self) -> str:
        return "matrix_row_max"
    
    @property
    def differentiable(self) -> bool:
        return False  # Not differentiable at equality points


class MatrixColMaxOp(Operation):
    """Max across columns (axis=1): M → vector"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.VECTOR
    
    def execute(self, m: np.ndarray) -> np.ndarray:
        return np.max(m, axis=1)
    
    @property
    def name(self) -> str:
        return "matrix_col_max"
    
    @property
    def differentiable(self) -> bool:
        return False  # Not differentiable at equality points


class MatrixFlattenOp(Operation):
    """Flatten matrix to vector"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.VECTOR
    
    def execute(self, m: np.ndarray) -> np.ndarray:
        return m.flatten()
    
    @property
    def name(self) -> str:
        return "matrix_flatten"
    
    @property
    def differentiable(self) -> bool:
        return True


class MatrixTransposeOp(Operation):
    """Transpose matrix: M^T"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, m: np.ndarray) -> np.ndarray:
        return m.T
    
    @property
    def name(self) -> str:
        return "matrix_transpose"
    
    @property
    def differentiable(self) -> bool:
        return True


# ==================== CROSS-TYPE OPERATIONS ====================

class ScalarVectorMulOp(Operation):
    """Multiply vector by scalar: s * v → vector"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR, MemoryType.VECTOR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.VECTOR
    
    def execute(self, s: float, v: np.ndarray) -> np.ndarray:
        return s * v
    
    @property
    def name(self) -> str:
        return "scalar_vector_mul"
    
    @property
    def differentiable(self) -> bool:
        return True


class ScalarVectorAddOp(Operation):
    """Add scalar to all vector elements: s + v → vector"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR, MemoryType.VECTOR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.VECTOR
    
    def execute(self, s: float, v: np.ndarray) -> np.ndarray:
        return s + v
    
    @property
    def name(self) -> str:
        return "scalar_vector_add"
    
    @property
    def differentiable(self) -> bool:
        return True


class ScalarMatrixMulOp(Operation):
    """Multiply matrix by scalar: s * M → matrix"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR, MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, s: float, m: np.ndarray) -> np.ndarray:
        return s * m
    
    @property
    def name(self) -> str:
        return "scalar_matrix_mul"
    
    @property
    def differentiable(self) -> bool:
        return True


class ScalarMatrixAddOp(Operation):
    """Add scalar to all matrix elements: s + M → matrix"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR, MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, s: float, m: np.ndarray) -> np.ndarray:
        return s + m
    
    @property
    def name(self) -> str:
        return "scalar_matrix_add"
    
    @property
    def differentiable(self) -> bool:
        return True

# ========== AUTOML ZERO OPS - ARITHMETIC OPERATIONS ==========

class AutoMLScalarAddOp(Operation):
    """OP1: s2=s3+s0 → sc = sa + sb"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR, MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, a: float, b: float) -> float:
        return a + b
    
    @property
    def name(self) -> str:
        return "automl_scalar_add"
    
    @property
    def differentiable(self) -> bool:
        return True


class AutoMLScalarSubOp(Operation):
    """OP2: s4=s0-s1 → sc = sa - sb"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR, MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, a: float, b: float) -> float:
        return a - b
    
    @property
    def name(self) -> str:
        return "automl_scalar_sub"
    
    @property
    def differentiable(self) -> bool:
        return True


class AutoMLScalarMulOp(Operation):
    """OP3: s8=s5*s5 → sc = sa × sb"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR, MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, a: float, b: float) -> float:
        return a * b
    
    @property
    def name(self) -> str:
        return "automl_scalar_mul"
    
    @property
    def differentiable(self) -> bool:
        return True


class AutoMLScalarDivOp(Operation):
    """OP4: s7=s5/s2 → sc = sa/sb (protected division)"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR, MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, a: float, b: float) -> float:
        if abs(b) < 1e-8:
            return 0.0
        return a / b
    
    @property
    def name(self) -> str:
        return "automl_scalar_div"
    
    @property
    def differentiable(self) -> bool:
        return False  # Has discontinuity at b=0


class AutoMLScalarAbsOp(Operation):
    """OP5: s8=abs(s0) → sb = |sa|"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, a: float) -> float:
        return abs(a)
    
    @property
    def name(self) -> str:
        return "automl_scalar_abs"
    
    @property
    def differentiable(self) -> bool:
        return False  # Not differentiable at a=0


class AutoMLScalarReciprocalOp(Operation):
    """OP6: s4=1/s8 → sb = 1/sa (protected reciprocal)"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, a: float) -> float:
        if abs(a) < 1e-8:
            return 0.0
        return 1.0 / a
    
    @property
    def name(self) -> str:
        return "automl_scalar_reciprocal"
    
    @property
    def differentiable(self) -> bool:
        return False  # Has discontinuity at a=0

# ========== AUTOML ZERO OPS - TRIGONOMETRIC OPERATIONS ==========

class AutoMLScalarSinOp(Operation):
    """OP7: s5=sin(s4) → sb = sin(sa)"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, a: float) -> float:
        # Handle NaN and infinity inputs
        if not np.isfinite(a):
            return 0.0
        return float(np.sin(a))
    
    @property
    def name(self) -> str:
        return "automl_scalar_sin"
    
    @property
    def differentiable(self) -> bool:
        return True


class AutoMLScalarCosOp(Operation):
    """OP8: s1=cos(s4) → sb = cos(sa)"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, a: float) -> float:
        # Handle NaN and infinity inputs
        if not np.isfinite(a):
            return 0.0
        return float(np.cos(a))
    
    @property
    def name(self) -> str:
        return "automl_scalar_cos"
    
    @property
    def differentiable(self) -> bool:
        return True


class AutoMLScalarTanOp(Operation):
    """OP9: s3=tan(s3) → sb = tan(sa)"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, a: float) -> float:
        # Handle NaN and infinity inputs
        if not np.isfinite(a):
            return 0.0
        return float(np.tan(a))
    
    @property
    def name(self) -> str:
        return "automl_scalar_tan"
    
    @property
    def differentiable(self) -> bool:
        return True


class AutoMLScalarArcsinOp(Operation):
    """OP10: s0=arcsin(s4) → sb = arcsin(sa)"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, a: float) -> float:
        # Clamp input to [-1, 1] before applying function
        a_clamped = np.clip(a, -1.0, 1.0)
        return float(np.arcsin(a_clamped))
    
    @property
    def name(self) -> str:
        return "automl_scalar_arcsin"
    
    @property
    def differentiable(self) -> bool:
        return True


class AutoMLScalarArccosOp(Operation):
    """OP11: s2=arccos(s0) → sb = arccos(sa)"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, a: float) -> float:
        # Clamp input to [-1, 1] before applying function
        a_clamped = np.clip(a, -1.0, 1.0)
        return float(np.arccos(a_clamped))
    
    @property
    def name(self) -> str:
        return "automl_scalar_arccos"
    
    @property
    def differentiable(self) -> bool:
        return True


class AutoMLScalarArctanOp(Operation):
    """OP12: s4=arctan(s0) → sb = arctan(sa)"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, a: float) -> float:
        return float(np.arctan(a))
    
    @property
    def name(self) -> str:
        return "automl_scalar_arctan"
    
    @property
    def differentiable(self) -> bool:
        return True

# ========== AUTOML ZERO OPS - PRE-CALCULUS OPERATIONS ==========

class AutoMLScalarExpOp(Operation):
    """OP13: s1=exp(s2) → sb = e^sa"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, a: float) -> float:
        # Clamp input to prevent overflow: exp(700) ≈ 1e304, exp(709) overflows
        if not np.isfinite(a):
            return 0.0
        a_clamped = np.clip(a, -700.0, 700.0)
        return float(np.exp(a_clamped))
    
    @property
    def name(self) -> str:
        return "automl_scalar_exp"
    
    @property
    def differentiable(self) -> bool:
        return True


class AutoMLScalarLogOp(Operation):
    """OP14: s0=log(s3) → sb = log(sa)"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, a: float) -> float:
        # Prevent log of non-positive - clamp input to max(1e-8, input)
        a_clamped = max(1e-8, a)
        return float(np.log(a_clamped))
    
    @property
    def name(self) -> str:
        return "automl_scalar_log"
    
    @property
    def differentiable(self) -> bool:
        return True


class AutoMLScalarHeavisideOp(Operation):
    """OP15: s3=heaviside(s0) → sb = 𝟙ℝ+(sa)"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, a: float) -> float:
        # Returns 1 if x ≥ 0, else 0
        return 1.0 if a >= 0.0 else 0.0
    
    @property
    def name(self) -> str:
        return "automl_scalar_heaviside"
    
    @property
    def differentiable(self) -> bool:
        return False  # Step function


class AutoMLVectorHeavisideOp(Operation):
    """OP16: v2=heaviside(v2) → vb(i) = 𝟙ℝ+(va(i)) ∀i"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.VECTOR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.VECTOR
    
    def execute(self, v: np.ndarray) -> np.ndarray:
        # Returns 1 if x ≥ 0, else 0 for each element
        return np.where(v >= 0.0, 1.0, 0.0).astype(np.float32)
    
    @property
    def name(self) -> str:
        return "automl_vector_heaviside"
    
    @property
    def differentiable(self) -> bool:
        return False  # Step function


class AutoMLMatrixHeavisideOp(Operation):
    """OP17: m7=heaviside(m3) → Mb(i,j) = 𝟙ℝ+(Ma(i,j)) ∀i,j"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, m: np.ndarray) -> np.ndarray:
        # Returns 1 if x ≥ 0, else 0 for each element
        return np.where(m >= 0.0, 1.0, 0.0).astype(np.float32)
    
    @property
    def name(self) -> str:
        return "automl_matrix_heaviside"
    
    @property
    def differentiable(self) -> bool:
        return False  # Step function

# ========== AUTOML ZERO OPS - LINEAR ALGEBRA - VECTOR OPERATIONS ==========

class AutoMLScalarVectorMulOp(Operation):
    """OP18: v1=s7*v1 → vc = sa × vb"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR, MemoryType.VECTOR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.VECTOR
    
    def execute(self, s: float, v: np.ndarray) -> np.ndarray:
        if not np.isfinite(s):
            s = 0.0
        s = float(np.clip(s, -SAFE_FLOAT_RANGE, SAFE_FLOAT_RANGE))
        v = _clip_array(v)
        result = (s * v).astype(np.float32)
        return _clip_array(result)
    
    @property
    def name(self) -> str:
        return "automl_scalar_vector_mul"
    
    @property
    def differentiable(self) -> bool:
        return True


class AutoMLScalarBroadcastOp(Operation):
    """OP19: v1=bcast(s3) → vb(i) = sa ∀i"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.VECTOR
    
    def execute(self, s: float) -> np.ndarray:
        # Note: This operation needs to know the vector size
        # For now, we'll return a default-sized vector (size 10)
        # In practice, this should be determined by the memory bank configuration
        return np.full(10, s, dtype=np.float32)
    
    @property
    def name(self) -> str:
        return "automl_scalar_broadcast"
    
    @property
    def differentiable(self) -> bool:
        return True


class AutoMLVectorReciprocalOp(Operation):
    """OP20: v5=1/v7 → vb(i) = 1/va(i) ∀i (protected)"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.VECTOR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.VECTOR
    
    def execute(self, v: np.ndarray) -> np.ndarray:
        # Prevent divide-by-zero - return 0 if denominator is zero
        result = np.zeros_like(v, dtype=np.float32)
        mask = np.abs(v) >= 1e-8
        result[mask] = 1.0 / v[mask]
        return result
    
    @property
    def name(self) -> str:
        return "automl_vector_reciprocal"
    
    @property
    def differentiable(self) -> bool:
        return False  # Has discontinuity at zero elements


class AutoMLVectorNormOp(Operation):
    """OP21: s0=norm(v3) → sb = ||va|| (L2 norm)"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.VECTOR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, v: np.ndarray) -> float:
        v = _clip_array(v)
        result = np.linalg.norm(v)
        if not np.isfinite(result):
            result = 0.0
        return float(result)
    
    @property
    def name(self) -> str:
        return "automl_vector_norm"
    
    @property
    def differentiable(self) -> bool:
        return True  # Differentiable everywhere except at v=0 (but commonly used)


class AutoMLVectorAbsOp(Operation):
    """OP22: v3=abs(v3) → vb(i) = |va(i)| ∀i"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.VECTOR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.VECTOR
    
    def execute(self, v: np.ndarray) -> np.ndarray:
        return np.abs(v).astype(np.float32)
    
    @property
    def name(self) -> str:
        return "automl_vector_abs"
    
    @property
    def differentiable(self) -> bool:
        return False  # Not differentiable at zero elements


class AutoMLVectorAddOp(Operation):
    """OP23: v5=v0+v9 → vc = va + vb"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.VECTOR, MemoryType.VECTOR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.VECTOR
    
    def execute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = _clip_array(a)
        b = _clip_array(b)
        result = (a + b).astype(np.float32)
        return _clip_array(result)
    
    @property
    def name(self) -> str:
        return "automl_vector_add"
    
    @property
    def differentiable(self) -> bool:
        return True


class AutoMLVectorSubOp(Operation):
    """OP24: v1=v0-v9 → vc = va - vb"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.VECTOR, MemoryType.VECTOR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.VECTOR
    
    def execute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = _clip_array(a)
        b = _clip_array(b)
        result = (a - b).astype(np.float32)
        result = _clip_array(result)
        result = np.nan_to_num(result, nan=0.0, posinf=SAFE_FLOAT_RANGE, neginf=-SAFE_FLOAT_RANGE)
        return result
    
    @property
    def name(self) -> str:
        return "automl_vector_sub"
    
    @property
    def differentiable(self) -> bool:
        return True


class AutoMLVectorMulOp(Operation):
    """OP25: v8=v1*v9 → vc(i) = va(i) × vb(i) ∀i"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.VECTOR, MemoryType.VECTOR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.VECTOR
    
    def execute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = _clip_array(a)
        b = _clip_array(b)
        result = (a * b).astype(np.float32)
        return _clip_array(result)
    
    @property
    def name(self) -> str:
        return "automl_vector_mul"
    
    @property
    def differentiable(self) -> bool:
        return True


class AutoMLVectorDivOp(Operation):
    """OP26: v9=v8/v2 → vc(i) = va(i)/vb(i) ∀i (protected)"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.VECTOR, MemoryType.VECTOR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.VECTOR
    
    def execute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # Prevent divide-by-zero - return 0 if denominator is zero
        a = _clip_array(a)
        b = _clip_array(b)
        result = np.zeros_like(a, dtype=np.float32)
        mask = np.abs(b) >= 1e-8
        if np.any(mask):
            safe_vals = (a[mask] / b[mask]).astype(np.float32)
            result[mask] = _clip_array(safe_vals)
        return result
    
    @property
    def name(self) -> str:
        return "automl_vector_div"
    
    @property
    def differentiable(self) -> bool:
        return False  # Has discontinuity at zero elements


class AutoMLVectorDotOp(Operation):
    """OP27: s6=dot(v1,v5) → sc = va^T × vb"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.VECTOR, MemoryType.VECTOR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, a: np.ndarray, b: np.ndarray) -> float:
        a = _clip_array(a)
        b = _clip_array(b)
        result = np.dot(a, b)
        if not np.isfinite(result):
            result = 0.0
        return float(result)
    
    @property
    def name(self) -> str:
        return "automl_vector_dot"
    
    @property
    def differentiable(self) -> bool:
        return True

# ========== AUTOML ZERO OPS - LINEAR ALGEBRA - MATRIX OPERATIONS ==========

class AutoMLVectorOuterOp(Operation):
    """OP28: m1=outer(v6,v5) → Mc = va × vb^T"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.VECTOR, MemoryType.VECTOR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.outer(a, b).astype(np.float32)
    
    @property
    def name(self) -> str:
        return "automl_vector_outer"
    
    @property
    def differentiable(self) -> bool:
        return True


class AutoMLScalarMatrixMulOp(Operation):
    """OP29: m1=s4*m2 → Mc = sa × Mb"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR, MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, s: float, m: np.ndarray) -> np.ndarray:
        return (s * m).astype(np.float32)
    
    @property
    def name(self) -> str:
        return "automl_scalar_matrix_mul"
    
    @property
    def differentiable(self) -> bool:
        return True


class AutoMLMatrixReciprocalOp(Operation):
    """OP30: m3=1/m0 → Mb(i,j) = 1/Ma(i,j) ∀i,j (protected)"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, m: np.ndarray) -> np.ndarray:
        # Prevent divide-by-zero - return 0 if denominator is zero
        result = np.zeros_like(m, dtype=np.float32)
        mask = np.abs(m) >= 1e-8
        result[mask] = 1.0 / m[mask]
        return result
    
    @property
    def name(self) -> str:
        return "automl_matrix_reciprocal"
    
    @property
    def differentiable(self) -> bool:
        return False  # Has discontinuity at zero elements


class AutoMLMatrixVectorDotOp(Operation):
    """OP31: v6=dot(m1,v0) → vc = Ma × vb"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX, MemoryType.VECTOR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.VECTOR
    
    def execute(self, m: np.ndarray, v: np.ndarray) -> np.ndarray:
        m = _clip_array(m)
        v = _clip_array(v)
        result = np.dot(m, v).astype(np.float32)
        result = _clip_array(result)
        result = np.nan_to_num(result, nan=0.0, posinf=SAFE_FLOAT_RANGE, neginf=-SAFE_FLOAT_RANGE)
        return result
    
    @property
    def name(self) -> str:
        return "automl_matrix_vector_dot"
    
    @property
    def differentiable(self) -> bool:
        return True


class AutoMLVectorBroadcastAxis0Op(Operation):
    """OP32: m2=bcast(v0,axis=0) → Mb(i,j) = va(i) ∀i,j"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.VECTOR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, v: np.ndarray) -> np.ndarray:
        # Broadcast vector along axis 0 (rows)
        # Note: This needs to know matrix shape, defaulting to (v.size, v.size)
        # In practice, this should be determined by the memory bank configuration
        n = v.size
        return np.tile(v.reshape(-1, 1), (1, n)).astype(np.float32)
    
    @property
    def name(self) -> str:
        return "automl_vector_broadcast_axis0"
    
    @property
    def differentiable(self) -> bool:
        return True


class AutoMLVectorBroadcastAxis1Op(Operation):
    """OP33: m2=bcast(v0,axis=1) → Mb(j,i) = va(i) ∀i,j"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.VECTOR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, v: np.ndarray) -> np.ndarray:
        # Broadcast vector along axis 1 (columns)
        # Note: This needs to know matrix shape, defaulting to (v.size, v.size)
        # In practice, this should be determined by the memory bank configuration
        n = v.size
        return np.tile(v.reshape(1, -1), (n, 1)).astype(np.float32)
    
    @property
    def name(self) -> str:
        return "automl_vector_broadcast_axis1"
    
    @property
    def differentiable(self) -> bool:
        return True


class AutoMLMatrixNormOp(Operation):
    """OP34: s2=norm(m1) → sb = ||Ma|| (L2 norm)"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, m: np.ndarray) -> float:
        m = _clip_array(m)
        result = np.linalg.norm(m)
        if not np.isfinite(result):
            result = 0.0
        return float(result)
    
    @property
    def name(self) -> str:
        return "automl_matrix_norm"
    
    @property
    def differentiable(self) -> bool:
        return True  # Differentiable everywhere except at m=0 (but commonly used)


class AutoMLMatrixNormAxis0Op(Operation):
    """OP35: v4=norm(m7,axis=0) → vb(i) = |Ma(i,:)| ∀i"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.VECTOR
    
    def execute(self, m: np.ndarray) -> np.ndarray:
        m = _clip_array(m)
        result = np.linalg.norm(m, axis=0).astype(np.float32)
        result = _clip_array(result)
        result = np.nan_to_num(result, nan=0.0, posinf=SAFE_FLOAT_RANGE, neginf=-SAFE_FLOAT_RANGE)
        return result
    
    @property
    def name(self) -> str:
        return "automl_matrix_norm_axis0"
    
    @property
    def differentiable(self) -> bool:
        return True


class AutoMLMatrixNormAxis1Op(Operation):
    """OP36: v4=norm(m7,axis=1) → vb(j) = |Ma(:,j)| ∀j"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.VECTOR
    
    def execute(self, m: np.ndarray) -> np.ndarray:
        m = _clip_array(m)
        result = np.linalg.norm(m, axis=1).astype(np.float32)
        result = _clip_array(result)
        result = np.nan_to_num(result, nan=0.0, posinf=SAFE_FLOAT_RANGE, neginf=-SAFE_FLOAT_RANGE)
        return result
    
    @property
    def name(self) -> str:
        return "automl_matrix_norm_axis1"
    
    @property
    def differentiable(self) -> bool:
        return True


class AutoMLMatrixTransposeOp(Operation):
    """OP37: m9=transpose(m3) → Mb = [Ma^T]"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, m: np.ndarray) -> np.ndarray:
        return m.T.astype(np.float32)
    
    @property
    def name(self) -> str:
        return "automl_matrix_transpose"
    
    @property
    def differentiable(self) -> bool:
        return True


class AutoMLMatrixAbsOp(Operation):
    """OP38: m1=abs(m8) → Mb(i,j) = |Ma(i,j)| ∀i,j"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, m: np.ndarray) -> np.ndarray:
        return np.abs(m).astype(np.float32)
    
    @property
    def name(self) -> str:
        return "automl_matrix_abs"
    
    @property
    def differentiable(self) -> bool:
        return False  # Not differentiable at zero elements


class AutoMLMatrixAddOp(Operation):
    """OP39: m2=m2+m0 → Mc = Ma + Mb"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX, MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a + b).astype(np.float32)
    
    @property
    def name(self) -> str:
        return "automl_matrix_add"
    
    @property
    def differentiable(self) -> bool:
        return True


class AutoMLMatrixSubOp(Operation):
    """OP40: m2=m3+m1 → Mc = Ma - Mb"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX, MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = _clip_array(a)
        b = _clip_array(b)
        result = (a - b).astype(np.float32)
        result = _clip_array(result)
        result = np.nan_to_num(result, nan=0.0, posinf=SAFE_FLOAT_RANGE, neginf=-SAFE_FLOAT_RANGE)
        return result
    
    @property
    def name(self) -> str:
        return "automl_matrix_sub"
    
    @property
    def differentiable(self) -> bool:
        return True


class AutoMLMatrixMulOp(Operation):
    """OP41: m3=m2*m3 → Mc(i,j) = Ma(i,j) × Mb(i,j) ∀i,j"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX, MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # Clamp inputs to prevent overflow
        a = _clip_array(a)
        b = _clip_array(b)
        result = (a * b).astype(np.float32)
        # Clamp result and handle NaN/inf
        result = _clip_array(result)
        result = np.nan_to_num(result, nan=0.0, posinf=SAFE_FLOAT_RANGE, neginf=-SAFE_FLOAT_RANGE)
        return result
    
    @property
    def name(self) -> str:
        return "automl_matrix_mul"
    
    @property
    def differentiable(self) -> bool:
        return True


class AutoMLMatrixDivOp(Operation):
    """OP42: m4=m2/m4 → Mc(i,j) = Ma(i,j)/Mb(i,j) ∀i,j (protected)"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX, MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # Prevent divide-by-zero - return 0 if denominator is zero
        a = _clip_array(a)
        b = _clip_array(b)
        result = np.zeros_like(a, dtype=np.float32)
        mask = np.abs(b) >= 1e-8
        if np.any(mask):
            safe_vals = (a[mask] / b[mask]).astype(np.float32)
            result[mask] = _clip_array(safe_vals)
        return result
    
    @property
    def name(self) -> str:
        return "automl_matrix_div"
    
    @property
    def differentiable(self) -> bool:
        return False  # Has discontinuity at zero elements


class AutoMLMatrixMatmulOp(Operation):
    """OP43: m5=matmul(m5,m7) → Mc = Ma × Mb"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX, MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # Clamp inputs to prevent overflow
        a = _clip_array(a)
        b = _clip_array(b)
        result = np.matmul(a, b).astype(np.float32)
        # Clamp result and handle NaN/inf
        result = _clip_array(result)
        result = np.nan_to_num(result, nan=0.0, posinf=SAFE_FLOAT_RANGE, neginf=-SAFE_FLOAT_RANGE)
        return result
    
    @property
    def name(self) -> str:
        return "automl_matrix_matmul"
    
    @property
    def differentiable(self) -> bool:
        return True

# ========== AUTOML ZERO OPS - PROBABILITY AND STATISTICS - MIN/MAX ==========

class AutoMLScalarMinOp(Operation):
    """OP44: s1=minimum(s2,s3) → sc = min(sa, sb)"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR, MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, a: float, b: float) -> float:
        return min(a, b)
    
    @property
    def name(self) -> str:
        return "automl_scalar_min"
    
    @property
    def differentiable(self) -> bool:
        return False  # Not differentiable at a=b


class AutoMLVectorMinOp(Operation):
    """OP45: v4=minimum(v3,v9) → vc(i) = min(va(i), vb(i)) ∀i"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.VECTOR, MemoryType.VECTOR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.VECTOR
    
    def execute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.minimum(a, b).astype(np.float32)
    
    @property
    def name(self) -> str:
        return "automl_vector_min"
    
    @property
    def differentiable(self) -> bool:
        return False  # Not differentiable at equality points


class AutoMLMatrixMinOp(Operation):
    """OP46: m2=minimum(m2,m1) → Mc(i,j) = min(Ma(i,j), Mb(i,j)) ∀i,j"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX, MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.minimum(a, b).astype(np.float32)
    
    @property
    def name(self) -> str:
        return "automl_matrix_min"
    
    @property
    def differentiable(self) -> bool:
        return False  # Not differentiable at equality points


class AutoMLScalarMaxOp(Operation):
    """OP47: s8=maximum(s3,s6) → sc = max(sa, sb)"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR, MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, a: float, b: float) -> float:
        return max(a, b)
    
    @property
    def name(self) -> str:
        return "automl_scalar_max"
    
    @property
    def differentiable(self) -> bool:
        return False  # Not differentiable at a=b


class AutoMLVectorMaxOp(Operation):
    """OP48: v7=maximum(v3,v6) → vc(i) = max(va(i), vb(i)) ∀i"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.VECTOR, MemoryType.VECTOR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.VECTOR
    
    def execute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.maximum(a, b).astype(np.float32)
    
    @property
    def name(self) -> str:
        return "automl_vector_max"
    
    @property
    def differentiable(self) -> bool:
        return False  # Not differentiable at equality points


class AutoMLMatrixMaxOp(Operation):
    """OP49: m7=maximum(m1,m0) → Mc(i,j) = max(Ma(i,j), Mb(i,j)) ∀i,j"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX, MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.maximum(a, b).astype(np.float32)
    
    @property
    def name(self) -> str:
        return "automl_matrix_max"
    
    @property
    def differentiable(self) -> bool:
        return False  # Not differentiable at equality points

# ========== AUTOML ZERO OPS - PROBABILITY AND STATISTICS - STATISTICAL OPERATIONS ==========

class AutoMLVectorMeanOp(Operation):
    """OP50: s2=mean(v2) → sb = mean(va)"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.VECTOR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, v: np.ndarray) -> float:
        v = _clip_array(v)
        result = np.mean(v)
        if not np.isfinite(result):
            result = 0.0
        return float(result)
    
    @property
    def name(self) -> str:
        return "automl_vector_mean"
    
    @property
    def differentiable(self) -> bool:
        return True


class AutoMLMatrixMeanOp(Operation):
    """OP51: s2=mean(m8) → sb = mean(Ma)"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, m: np.ndarray) -> float:
        m = _clip_array(m)
        result = np.mean(m)
        if not np.isfinite(result):
            result = 0.0
        return float(result)
    
    @property
    def name(self) -> str:
        return "automl_matrix_mean"
    
    @property
    def differentiable(self) -> bool:
        return True


class AutoMLMatrixMeanAxis0Op(Operation):
    """OP52: v1=mean(m2,axis=0) → vb(i) = mean(Ma(i,:)) ∀i"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.VECTOR
    
    def execute(self, m: np.ndarray) -> np.ndarray:
        m = _clip_array(m)
        result = np.mean(m, axis=0).astype(np.float32)
        result = _clip_array(result)
        result = np.nan_to_num(result, nan=0.0, posinf=SAFE_FLOAT_RANGE, neginf=-SAFE_FLOAT_RANGE)
        return result
    
    @property
    def name(self) -> str:
        return "automl_matrix_mean_axis0"
    
    @property
    def differentiable(self) -> bool:
        return True


class AutoMLMatrixStdAxis0Op(Operation):
    """OP53: v3=std(m2,axis=0) → vb(i) = stdev(Ma(i,:)) ∀i"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.VECTOR
    
    def execute(self, m: np.ndarray) -> np.ndarray:
        m = _clip_array(m)
        result = np.std(m, axis=0).astype(np.float32)
        result = _clip_array(result)
        result = np.nan_to_num(result, nan=0.0, posinf=SAFE_FLOAT_RANGE, neginf=-SAFE_FLOAT_RANGE)
        return result
    
    @property
    def name(self) -> str:
        return "automl_matrix_std_axis0"
    
    @property
    def differentiable(self) -> bool:
        return True


class AutoMLVectorStdOp(Operation):
    """OP54: s3=std(v3) → sb = stdev(va)"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.VECTOR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, v: np.ndarray) -> float:
        v = _clip_array(v)
        result = np.std(v)
        if not np.isfinite(result):
            result = 0.0
        return float(result)
    
    @property
    def name(self) -> str:
        return "automl_vector_std"
    
    @property
    def differentiable(self) -> bool:
        return True


class AutoMLMatrixStdOp(Operation):
    """OP55: s4=std(m0) → sb = stdev(Ma)"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, m: np.ndarray) -> float:
        m = _clip_array(m)
        result = np.std(m)
        if not np.isfinite(result):
            result = 0.0
        return float(result)
    
    @property
    def name(self) -> str:
        return "automl_matrix_std"
    
    @property
    def differentiable(self) -> bool:
        return True

# ========== AUTOML ZERO OPS - CONSTANT AND RANDOM INITIALIZATION ==========

class AutoMLScalarConstantOp(Operation):
    """OP56: s2=0.1 → sa = γ (takes constant as input)"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR]  # The constant value
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, gamma: float) -> float:
        return gamma
    
    @property
    def name(self) -> str:
        return "automl_scalar_constant"
    
    @property
    def differentiable(self) -> bool:
        return True


class AutoMLVectorElementConstantOp(Operation):
    """OP57: v3[5]=-2.4 → va(i) = γ (with modulo wrapping and copy)"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.VECTOR, MemoryType.SCALAR, MemoryType.SCALAR]  # vector, index, constant
    
    def output_type(self) -> MemoryType:
        return MemoryType.VECTOR
    
    def execute(self, v: np.ndarray, idx: float, gamma: float) -> np.ndarray:
        # Copy before modify
        result = v.copy()
        # Modulo wrapping for index - handle NaN/infinity
        if not np.isfinite(idx):
            idx_int = 0  # Default to first element if invalid
        else:
            idx_int = int(idx) % len(result)
        result[idx_int] = gamma
        return result.astype(np.float32)
    
    @property
    def name(self) -> str:
        return "automl_vector_element_constant"
    
    @property
    def differentiable(self) -> bool:
        return False  # Not differentiable due to indexing


class AutoMLMatrixElementConstantOp(Operation):
    """OP58: m2[5,1]=-0.03 → Ma(i,j) = γ (with modulo wrapping and copy)"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX, MemoryType.SCALAR, MemoryType.SCALAR, MemoryType.SCALAR]  # matrix, row, col, constant
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, m: np.ndarray, row: float, col: float, gamma: float) -> np.ndarray:
        # Copy before modify
        result = m.copy()
        # Modulo wrapping for both indices - handle NaN/infinity
        if not np.isfinite(row):
            row_int = 0  # Default to first row if invalid
        else:
            row_int = int(row) % result.shape[0]
        if not np.isfinite(col):
            col_int = 0  # Default to first column if invalid
        else:
            col_int = int(col) % result.shape[1]
        result[row_int, col_int] = gamma
        return result.astype(np.float32)
    
    @property
    def name(self) -> str:
        return "automl_matrix_element_constant"
    
    @property
    def differentiable(self) -> bool:
        return False  # Not differentiable due to indexing


class AutoMLScalarUniformOp(Operation):
    """OP59: s4=uniform(-1,1) → sa = U(α,β)"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR, MemoryType.SCALAR]  # min, max
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, min_val: float, max_val: float) -> float:
        # Handle NaN and inf
        if not np.isfinite(min_val):
            min_val = 0.0
        if not np.isfinite(max_val):
            max_val = 1.0
        # Swap if max < min, handle equal bounds
        if max_val < min_val:
            min_val, max_val = max_val, min_val
        # Clamp to reasonable range to prevent overflow
        MAX_RANGE = 1e6
        min_val = np.clip(min_val, -MAX_RANGE, MAX_RANGE)
        max_val = np.clip(max_val, -MAX_RANGE, MAX_RANGE)
        if abs(max_val - min_val) < 1e-8:
            return min_val
        # Respect global seed via np.random
        return float(np.random.uniform(min_val, max_val))
    
    @property
    def name(self) -> str:
        return "automl_scalar_uniform"
    
    @property
    def differentiable(self) -> bool:
        return False  # Non-deterministic


class AutoMLVectorUniformOp(Operation):
    """OP60: v1=uniform(0.4,0.8) → va(i) = U(α,β) ∀i"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR, MemoryType.SCALAR]  # min, max
    
    def output_type(self) -> MemoryType:
        return MemoryType.VECTOR
    
    def execute(self, min_val: float, max_val: float) -> np.ndarray:
        # Handle NaN and inf
        if not np.isfinite(min_val):
            min_val = 0.0
        if not np.isfinite(max_val):
            max_val = 1.0
        # Swap if max < min, handle equal bounds
        if max_val < min_val:
            min_val, max_val = max_val, min_val
        # Clamp to reasonable range to prevent overflow
        MAX_RANGE = 1e6
        min_val = np.clip(min_val, -MAX_RANGE, MAX_RANGE)
        max_val = np.clip(max_val, -MAX_RANGE, MAX_RANGE)
        # Note: This needs to know vector size, defaulting to 10
        # In practice, this should be determined by the memory bank configuration
        size = 10
        if abs(max_val - min_val) < 1e-8:
            return np.full(size, min_val, dtype=np.float32)
        # Respect global seed via np.random
        return np.random.uniform(min_val, max_val, size=size).astype(np.float32)
    
    @property
    def name(self) -> str:
        return "automl_vector_uniform"
    
    @property
    def differentiable(self) -> bool:
        return False  # Non-deterministic


class AutoMLMatrixUniformOp(Operation):
    """OP61: m0=uniform(-0.5,0.6) → Ma(i,j) = U(α,β) ∀i,j"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR, MemoryType.SCALAR]  # min, max
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, min_val: float, max_val: float) -> np.ndarray:
        # Handle NaN and inf
        if not np.isfinite(min_val):
            min_val = 0.0
        if not np.isfinite(max_val):
            max_val = 1.0
        # Swap if max < min, handle equal bounds
        if max_val < min_val:
            min_val, max_val = max_val, min_val
        # Clamp to reasonable range to prevent overflow
        MAX_RANGE = 1e6
        min_val = np.clip(min_val, -MAX_RANGE, MAX_RANGE)
        max_val = np.clip(max_val, -MAX_RANGE, MAX_RANGE)
        # Note: This needs to know matrix shape, defaulting to (10, 10)
        # In practice, this should be determined by the memory bank configuration
        shape = (10, 10)
        if abs(max_val - min_val) < 1e-8:
            return np.full(shape, min_val, dtype=np.float32)
        # Respect global seed via np.random
        return np.random.uniform(min_val, max_val, size=shape).astype(np.float32)
    
    @property
    def name(self) -> str:
        return "automl_matrix_uniform"
    
    @property
    def differentiable(self) -> bool:
        return False  # Non-deterministic


class AutoMLScalarGaussianOp(Operation):
    """OP62: s4=gaussian(0.1,0.7) → sa = N(μ,σ)"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR, MemoryType.SCALAR]  # mean, std
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, mean: float, std: float) -> float:
        # Handle NaN and inf
        if not np.isfinite(mean):
            mean = 0.0
        if not np.isfinite(std):
            std = 1.0
        # Clamp to reasonable range
        MAX_RANGE = 1e6
        mean = np.clip(mean, -MAX_RANGE, MAX_RANGE)
        # Ensure std > 0 (use 1e-8 minimum) and clamp
        std = max(1e-8, min(abs(std), MAX_RANGE))
        # Respect global seed via np.random
        return float(np.random.normal(mean, std))
    
    @property
    def name(self) -> str:
        return "automl_scalar_gaussian"
    
    @property
    def differentiable(self) -> bool:
        return False  # Non-deterministic


class AutoMLVectorGaussianOp(Operation):
    """OP63: v8=gaussian(0.4,1) → va(i) = N(μ,σ) ∀i"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR, MemoryType.SCALAR]  # mean, std
    
    def output_type(self) -> MemoryType:
        return MemoryType.VECTOR
    
    def execute(self, mean: float, std: float) -> np.ndarray:
        # Handle NaN and inf
        if not np.isfinite(mean):
            mean = 0.0
        if not np.isfinite(std):
            std = 1.0
        # Clamp to reasonable range
        MAX_RANGE = 1e6
        mean = np.clip(mean, -MAX_RANGE, MAX_RANGE)
        # Ensure std > 0 (use 1e-8 minimum) and clamp
        std = max(1e-8, min(abs(std), MAX_RANGE))
        # Note: This needs to know vector size, defaulting to 10
        # In practice, this should be determined by the memory bank configuration
        size = 10
        # Respect global seed via np.random
        return np.random.normal(mean, std, size=size).astype(np.float32)
    
    @property
    def name(self) -> str:
        return "automl_vector_gaussian"
    
    @property
    def differentiable(self) -> bool:
        return False  # Non-deterministic


class AutoMLMatrixGaussianOp(Operation):
    """OP64: m2=gaussian(-2,1.3) → Ma(i,j) = N(μ,σ) ∀i,j"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR, MemoryType.SCALAR]  # mean, std
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, mean: float, std: float) -> np.ndarray:
        # Handle NaN and inf
        if not np.isfinite(mean):
            mean = 0.0
        if not np.isfinite(std):
            std = 1.0
        # Clamp to reasonable range
        MAX_RANGE = 1e6
        mean = np.clip(mean, -MAX_RANGE, MAX_RANGE)
        # Ensure std > 0 (use 1e-8 minimum) and clamp
        std = max(1e-8, min(abs(std), MAX_RANGE))
        # Note: This needs to know matrix shape, defaulting to (10, 10)
        # In practice, this should be determined by the memory bank configuration
        shape = (10, 10)
        # Respect global seed via np.random
        return np.random.normal(mean, std, size=shape).astype(np.float32)
    
    @property
    def name(self) -> str:
        return "automl_matrix_gaussian"
    
    @property
    def differentiable(self) -> bool:
        return False  # Non-deterministic

# ==================== AUTOML REGISTRY ====================

# AutoML Arithmetic Operations
AUTOML_ARITHMETIC_OPS = [
    AutoMLScalarAddOp,
    AutoMLScalarSubOp,
    AutoMLScalarMulOp,
    AutoMLScalarDivOp,
    AutoMLScalarAbsOp,
    AutoMLScalarReciprocalOp,
]

# AutoML Trigonometric Operations
AUTOML_TRIGONOMETRIC_OPS = [
    AutoMLScalarSinOp,
    AutoMLScalarCosOp,
    AutoMLScalarTanOp,
    AutoMLScalarArcsinOp,
    AutoMLScalarArccosOp,
    AutoMLScalarArctanOp,
]

# AutoML Pre-Calculus Operations
AUTOML_PRECALC_OPS = [
    AutoMLScalarExpOp,
    AutoMLScalarLogOp,
    AutoMLScalarHeavisideOp,
    AutoMLVectorHeavisideOp,
    AutoMLMatrixHeavisideOp,
]

# AutoML Linear Algebra - Vector Operations
AUTOML_VECTOR_OPS = [
    AutoMLScalarVectorMulOp,
    AutoMLScalarBroadcastOp,
    AutoMLVectorReciprocalOp,
    AutoMLVectorNormOp,
    AutoMLVectorAbsOp,
    AutoMLVectorAddOp,
    AutoMLVectorSubOp,
    AutoMLVectorMulOp,
    AutoMLVectorDivOp,
    AutoMLVectorDotOp,
]

# AutoML Linear Algebra - Matrix Operations
AUTOML_MATRIX_OPS = [
    AutoMLVectorOuterOp,
    AutoMLScalarMatrixMulOp,
    AutoMLMatrixReciprocalOp,
    AutoMLMatrixVectorDotOp,
    AutoMLVectorBroadcastAxis0Op,
    AutoMLVectorBroadcastAxis1Op,
    AutoMLMatrixNormOp,
    AutoMLMatrixNormAxis0Op,
    AutoMLMatrixNormAxis1Op,
    AutoMLMatrixTransposeOp,
    AutoMLMatrixAbsOp,
    AutoMLMatrixAddOp,
    AutoMLMatrixSubOp,
    AutoMLMatrixMulOp,
    AutoMLMatrixDivOp,
    AutoMLMatrixMatmulOp,
]

# AutoML Min/Max Operations
AUTOML_MINMAX_OPS = [
    AutoMLScalarMinOp,
    AutoMLVectorMinOp,
    AutoMLMatrixMinOp,
    AutoMLScalarMaxOp,
    AutoMLVectorMaxOp,
    AutoMLMatrixMaxOp,
]

# AutoML Statistical Operations
AUTOML_STATISTICAL_OPS = [
    AutoMLVectorMeanOp,
    AutoMLMatrixMeanOp,
    AutoMLMatrixMeanAxis0Op,
    AutoMLMatrixStdAxis0Op,
    AutoMLVectorStdOp,
    AutoMLMatrixStdOp,
]

# AutoML Constant and Random Initialization Operations
AUTOML_CONSTANT_RANDOM_OPS = [
    AutoMLScalarConstantOp,
    AutoMLVectorElementConstantOp,
    AutoMLMatrixElementConstantOp,
    AutoMLScalarUniformOp,
    AutoMLVectorUniformOp,
    AutoMLMatrixUniformOp,
    AutoMLScalarGaussianOp,
    AutoMLVectorGaussianOp,
    AutoMLMatrixGaussianOp,
]

# All AutoML operations combined
AUTOML_ALL_OPS = (
    AUTOML_ARITHMETIC_OPS +
    AUTOML_TRIGONOMETRIC_OPS +
    AUTOML_PRECALC_OPS +
    AUTOML_VECTOR_OPS +
    AUTOML_MATRIX_OPS +
    AUTOML_MINMAX_OPS +
    AUTOML_STATISTICAL_OPS +
    AUTOML_CONSTANT_RANDOM_OPS
)

# ========== CV OPERATIONS - EDGE DETECTION ==========

def _normalize_to_uint8(matrix: np.ndarray) -> np.ndarray:
    """Convert float32 matrix to uint8 [0, 255] for OpenCV operations"""
    if matrix.size == 0:
        return matrix.astype(np.uint8)
    # Normalize to [0, 255] range
    min_val = matrix.min()
    max_val = matrix.max()
    # Check for invalid values
    if not np.isfinite(min_val) or not np.isfinite(max_val):
        return np.full_like(matrix, 128, dtype=np.uint8)
    if abs(max_val - min_val) < 1e-8:
        return np.full_like(matrix, 128, dtype=np.uint8)
    normalized = ((matrix - min_val) / (max_val - min_val) * 255.0).clip(0, 255)
    return normalized.astype(np.uint8)


def _normalize_from_uint8(matrix: np.ndarray) -> np.ndarray:
    """Convert uint8 matrix back to float32 [0, 1] range"""
    return (matrix.astype(np.float32) / 255.0)


def _safe_int(value: float, min_val: int = 1, max_val: int = 255) -> int:
    """Safely convert a float to int, handling infinity and NaN.
    
    Args:
        value: The float value to convert
        min_val: Minimum allowed integer value
        max_val: Maximum allowed integer value
        
    Returns:
        An integer in the range [min_val, max_val]
    """
    if not np.isfinite(value):
        # Handle infinity and NaN by returning a default value
        return min_val
    # Clamp to valid range and convert to int
    return max(min_val, min(max_val, int(abs(value))))

def _ensure_odd_kernel_size(kernel_size: float) -> int:
    """Ensure kernel size is odd and positive"""
    k = _safe_int(kernel_size, min_val=1, max_val=255)
    if k % 2 == 0:
        k += 1
    return k


class CVSobelXOp(Operation):
    """sobel_x: Vertical edges (horizontal gradients)"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX, MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, m: np.ndarray, kernel_size: float) -> np.ndarray:
        
        if m.size == 0:
            return m.astype(np.float32)
        
        # Convert to uint8
        img_uint8 = _normalize_to_uint8(m)
        
        # Ensure odd kernel size
        ksize = _ensure_odd_kernel_size(kernel_size)
        ksize = min(ksize, 31)  # OpenCV limit
        
        # Apply Sobel X (vertical edges)
        sobel_x = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=ksize)
        
        # Convert back to float32 and normalize
        result = np.abs(sobel_x).astype(np.float32)
        if result.max() > 0:
            result = result / result.max()
        return result
    
    @property
    def name(self) -> str:
        return "cv_sobel_x"
    
    @property
    def differentiable(self) -> bool:
        return False  # Discrete operation


class CVSobelYOp(Operation):
    """sobel_y: Horizontal edges (vertical gradients)"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX, MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, m: np.ndarray, kernel_size: float) -> np.ndarray:
        
        if m.size == 0:
            return m.astype(np.float32)
        
        # Convert to uint8
        img_uint8 = _normalize_to_uint8(m)
        
        # Ensure odd kernel size
        ksize = _ensure_odd_kernel_size(kernel_size)
        ksize = min(ksize, 31)  # OpenCV limit
        
        # Apply Sobel Y (horizontal edges)
        sobel_y = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=ksize)
        
        # Convert back to float32 and normalize
        result = np.abs(sobel_y).astype(np.float32)
        if result.max() > 0:
            result = result / result.max()
        return result
    
    @property
    def name(self) -> str:
        return "cv_sobel_y"
    
    @property
    def differentiable(self) -> bool:
        return False  # Discrete operation


class CVCannyEdgesOp(Operation):
    """canny_edges: Multi-stage robust edge detector"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX, MemoryType.SCALAR, MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, m: np.ndarray, low_thresh: float, high_thresh: float) -> np.ndarray:
        if cv2 is None:
            raise ImportError("OpenCV (cv2) is required for CV operations")
        if m.size == 0:
            return m.astype(np.float32)
        
        # Convert to uint8
        img_uint8 = _normalize_to_uint8(m)
        
        # Ensure thresholds are valid (low < high, both positive)
        low = max(0, min(255, float(low_thresh)))
        high = max(0, min(255, float(high_thresh)))
        if low >= high:
            high = low + 1.0
        if high > 255:
            high = 255.0
            low = max(0, high - 1.0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(img_uint8, int(low), int(high))
        
        # Convert back to float32
        return _normalize_from_uint8(edges)
    
    @property
    def name(self) -> str:
        return "cv_canny_edges"
    
    @property
    def differentiable(self) -> bool:
        return False  # Discrete operation

# ========== CV OPERATIONS - FILTERING ==========

class CVGaussianBlurOp(Operation):
    """gaussian_blur: Smooth noise while preserving structure"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX, MemoryType.SCALAR, MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, m: np.ndarray, kernel_size: float, sigma: float) -> np.ndarray:
        if cv2 is None:
            raise ImportError("OpenCV (cv2) is required for CV operations")
        if m.size == 0:
            return m.astype(np.float32)
        
        # Convert to uint8
        img_uint8 = _normalize_to_uint8(m)
        
        # Ensure odd kernel size
        ksize = _ensure_odd_kernel_size(kernel_size)
        ksize = min(ksize, 31)  # OpenCV limit
        
        # Ensure positive sigma
        sigma_val = max(0.1, abs(float(sigma)))
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(img_uint8, (ksize, ksize), sigmaX=sigma_val, sigmaY=sigma_val)
        
        # Convert back to float32
        return _normalize_from_uint8(blurred)
    
    @property
    def name(self) -> str:
        return "cv_gaussian_blur"
    
    @property
    def differentiable(self) -> bool:
        return False  # Discrete operation

class CVResizeOp(Operation):
    """resize: Resize image to specified dimensions"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX, MemoryType.SCALAR, MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, m: np.ndarray, width: float, height: float) -> np.ndarray:
        if cv2 is None:
            raise ImportError("OpenCV (cv2) is required for CV operations")
        if m.size == 0:
            return m.astype(np.float32)
        
        # Convert to uint8
        img_uint8 = _normalize_to_uint8(m)
        
        # Ensure positive dimensions
        target_w = _safe_int(width, min_val=1, max_val=1000)
        target_h = _safe_int(height, min_val=1, max_val=1000)
        
        # Resize using INTER_AREA for downsampling, INTER_LINEAR for upsampling
        current_h, current_w = img_uint8.shape
        if target_w < current_w or target_h < current_h:
            # Downsampling: use INTER_AREA (best quality for downsampling)
            interpolation = cv2.INTER_AREA
        else:
            # Upsampling: use INTER_LINEAR (faster than INTER_CUBIC)
            interpolation = cv2.INTER_LINEAR
        
        resized = cv2.resize(img_uint8, (target_w, target_h), interpolation=interpolation)
        
        # Convert back to float32
        return _normalize_from_uint8(resized)
    
    @property
    def name(self) -> str:
        return "cv_resize"
    
    @property
    def differentiable(self) -> bool:
        return False  # Discrete operation

# ========== CV OPERATIONS - SEGMENTATION ==========

class CVAdaptiveThresholdOp(Operation):
    """adaptive_threshold: Local adaptive thresholding (handles lighting variations)"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX, MemoryType.SCALAR, MemoryType.SCALAR, MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, m: np.ndarray, max_val: float, block_size: float, C: float) -> np.ndarray:
        if cv2 is None:
            raise ImportError("OpenCV (cv2) is required for CV operations")
        if m.size == 0:
            return m.astype(np.float32)
        
        # Convert to uint8
        img_uint8 = _normalize_to_uint8(m)
        
        # Ensure max_val is in valid range
        max_val_int = _safe_int(max_val, min_val=1, max_val=255)
        
        # Ensure block_size is odd and positive
        block = _ensure_odd_kernel_size(block_size)
        block = max(3, min(block, 255))  # OpenCV requires >= 3
        
        # C parameter (subtracted from mean)
        C_val = float(C)
        
        # Apply adaptive threshold
        thresholded = cv2.adaptiveThreshold(
            img_uint8, 
            max_val_int, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            block, 
            C_val
        )
        
        # Convert back to float32
        return _normalize_from_uint8(thresholded)
    
    @property
    def name(self) -> str:
        return "cv_adaptive_threshold"
    
    @property
    def differentiable(self) -> bool:
        return False  # Discrete operation

# ========== CV OPERATIONS - MORPHOLOGICAL ==========

class CVErodeOp(Operation):
    """erode: Shrink bright regions"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX, MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, m: np.ndarray, kernel_size: float) -> np.ndarray:
        if cv2 is None:
            raise ImportError("OpenCV (cv2) is required for CV operations")
        if m.size == 0:
            return m.astype(np.float32)
        
        # Convert to uint8
        img_uint8 = _normalize_to_uint8(m)
        
        # Ensure positive kernel size
        ksize = _safe_int(kernel_size, min_val=1, max_val=255)
        kernel = np.ones((ksize, ksize), np.uint8)
        
        # Apply erosion
        eroded = cv2.erode(img_uint8, kernel, iterations=1)
        
        # Convert back to float32
        return _normalize_from_uint8(eroded)
    
    @property
    def name(self) -> str:
        return "cv_erode"
    
    @property
    def differentiable(self) -> bool:
        return False  # Discrete operation


class CVDilateOp(Operation):
    """dilate: Expand bright regions"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX, MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, m: np.ndarray, kernel_size: float) -> np.ndarray:
        if cv2 is None:
            raise ImportError("OpenCV (cv2) is required for CV operations")
        if m.size == 0:
            return m.astype(np.float32)
        
        # Convert to uint8
        img_uint8 = _normalize_to_uint8(m)
        
        # Ensure positive kernel size
        ksize = _safe_int(kernel_size, min_val=1, max_val=255)
        kernel = np.ones((ksize, ksize), np.uint8)
        
        # Apply dilation
        dilated = cv2.dilate(img_uint8, kernel, iterations=1)
        
        # Convert back to float32
        return _normalize_from_uint8(dilated)
    
    @property
    def name(self) -> str:
        return "cv_dilate"
    
    @property
    def differentiable(self) -> bool:
        return False  # Discrete operation

# ========== CV OPERATIONS - CONTRAST ENHANCEMENT ==========

class CVCLAHEOp(Operation):
    """clahe: Contrast Limited Adaptive Histogram Equalization"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX, MemoryType.SCALAR, MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, m: np.ndarray, clip_limit: float, tile_size: float) -> np.ndarray:
        if cv2 is None:
            raise ImportError("OpenCV (cv2) is required for CV operations")
        if m.size == 0:
            return m.astype(np.float32)
        
        # Convert to uint8
        img_uint8 = _normalize_to_uint8(m)
        
        # Ensure positive clip_limit
        clip = max(0.1, abs(float(clip_limit)))
        
        # Ensure positive tile_size (must be at least 1)
        tile = _safe_int(tile_size, min_val=1, max_val=255)
        
        # Create CLAHE object and apply
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
        enhanced = clahe.apply(img_uint8)
        
        # Convert back to float32
        return _normalize_from_uint8(enhanced)
    
    @property
    def name(self) -> str:
        return "cv_clahe"
    
    @property
    def differentiable(self) -> bool:
        return False  # Discrete operation

# ========== CV OPERATIONS - KEYPOINT DETECTION ==========

class CVHarrisCornersOp(Operation):
    """harris_corners: Corner/keypoint detection"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX, MemoryType.SCALAR, MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, m: np.ndarray, block_size: float, k: float) -> np.ndarray:
        if cv2 is None:
            raise ImportError("OpenCV (cv2) is required for CV operations")
        if m.size == 0:
            return m.astype(np.float32)
        
        # Convert to uint8
        img_uint8 = _normalize_to_uint8(m)
        
        # Ensure block_size is at least 2
        block = _safe_int(block_size, min_val=2, max_val=255)
        
        # k parameter (typically 0.04-0.06)
        k_val = float(k)
        
        # Apply Harris corner detection
        corners = cv2.cornerHarris(img_uint8, block, 3, k_val)
        
        # Normalize to [0, 1] range
        corners_normalized = np.zeros_like(corners, dtype=np.float32)
        if corners.max() > corners.min():
            corners_normalized = (corners - corners.min()) / (corners.max() - corners.min())
        
        return corners_normalized
    
    @property
    def name(self) -> str:
        return "cv_harris_corners"
    
    @property
    def differentiable(self) -> bool:
        return False  # Discrete operation

# ========== CV OPERATIONS - TEMPORAL/MOTION ==========

class CVFrameDifferenceOp(Operation):
    """frame_difference: Simple motion detection between frames"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX, MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        if prev.size == 0 or curr.size == 0:
            return np.zeros_like(prev if prev.size > 0 else curr, dtype=np.float32)
        
        # Ensure same shape
        if prev.shape != curr.shape:
            # Resize curr to match prev if possible
            min_h = min(prev.shape[0], curr.shape[0])
            min_w = min(prev.shape[1], curr.shape[1])
            prev = prev[:min_h, :min_w]
            curr = curr[:min_h, :min_w]
        
        # Compute absolute difference
        diff = np.abs(prev.astype(np.float32) - curr.astype(np.float32))
        
        # Normalize to [0, 1] if needed
        if diff.max() > 1.0:
            diff = diff / diff.max() if diff.max() > 0 else diff
        
        return diff.astype(np.float32)
    
    @property
    def name(self) -> str:
        return "cv_frame_difference"
    
    @property
    def differentiable(self) -> bool:
        return True  # Simple subtraction, differentiable

# ========== CV OPERATIONS - POOLING ==========

class CVMaxPoolOp(Operation):
    """max_pool: Downsample by taking max"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX, MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, m: np.ndarray, kernel_size: float) -> np.ndarray:
        if m.size == 0:
            return m.astype(np.float32)
        
        # Ensure positive kernel size
        ksize = _safe_int(kernel_size, min_val=1, max_val=255)
        
        if ndimage is not None:
            # Use scipy for max pooling
            # Apply maximum filter
            filtered = ndimage.maximum_filter(m.astype(np.float32), size=ksize)
            # Downsample by taking every ksize-th element
            pooled = filtered[::ksize, ::ksize]
        else:
            # Fallback: simple downsampling by taking max in each block
            h, w = m.shape
            new_h = h // ksize
            new_w = w // ksize
            if new_h == 0 or new_w == 0:
                return m.astype(np.float32)
            
            pooled = np.zeros((new_h, new_w), dtype=np.float32)
            for i in range(new_h):
                for j in range(new_w):
                    i_start = i * ksize
                    i_end = min(i_start + ksize, h)
                    j_start = j * ksize
                    j_end = min(j_start + ksize, w)
                    pooled[i, j] = m[i_start:i_end, j_start:j_end].max()
        
        return pooled.astype(np.float32)
    
    @property
    def name(self) -> str:
        return "cv_max_pool"
    
    @property
    def differentiable(self) -> bool:
        return False  # Discrete max operation


class CVAvgPoolOp(Operation):
    """avg_pool: Downsample by averaging"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX, MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, m: np.ndarray, kernel_size: float) -> np.ndarray:
        if m.size == 0:
            return m.astype(np.float32)
        
        # Ensure positive kernel size
        ksize = _safe_int(kernel_size, min_val=1, max_val=255)
        
        if ndimage is not None:
            # Use scipy for average pooling
            # Apply uniform filter (average)
            filtered = ndimage.uniform_filter(m.astype(np.float32), size=ksize)
            # Downsample by taking every ksize-th element
            pooled = filtered[::ksize, ::ksize]
        else:
            # Fallback: simple downsampling by averaging each block
            h, w = m.shape
            new_h = h // ksize
            new_w = w // ksize
            if new_h == 0 or new_w == 0:
                return m.astype(np.float32)
            
            pooled = np.zeros((new_h, new_w), dtype=np.float32)
            for i in range(new_h):
                for j in range(new_w):
                    i_start = i * ksize
                    i_end = min(i_start + ksize, h)
                    j_start = j * ksize
                    j_end = min(j_start + ksize, w)
                    pooled[i, j] = m[i_start:i_end, j_start:j_end].mean()
        
        return pooled.astype(np.float32)
    
    @property
    def name(self) -> str:
        return "cv_avg_pool"
    
    @property
    def differentiable(self) -> bool:
        return True  # Average is differentiable

# ==================== CV OPERATIONS REGISTRY ====================

# CV Edge Detection Operations
CV_EDGE_DETECTION_OPS = [
    CVSobelXOp,
    CVSobelYOp,
    CVCannyEdgesOp,
]

# CV Filtering Operations
CV_FILTERING_OPS = [
    CVGaussianBlurOp,
    CVResizeOp,
]

# CV Segmentation Operations
CV_SEGMENTATION_OPS = [
    CVAdaptiveThresholdOp,
]

# CV Morphological Operations
CV_MORPHOLOGICAL_OPS = [
    CVErodeOp,
    CVDilateOp,
]

# CV Contrast Enhancement Operations
CV_CONTRAST_OPS = [
    CVCLAHEOp,
]

# CV Keypoint Detection Operations
CV_KEYPOINT_OPS = [
    CVHarrisCornersOp,
]

# CV Temporal/Motion Operations
CV_TEMPORAL_OPS = [
    CVFrameDifferenceOp,
]

# CV Pooling Operations
CV_POOLING_OPS = [
    CVMaxPoolOp,
    CVAvgPoolOp,
]

# All CV operations combined
CV_ALL_OPS = (
    CV_EDGE_DETECTION_OPS +
    CV_FILTERING_OPS +
    CV_SEGMENTATION_OPS +
    CV_MORPHOLOGICAL_OPS +
    CV_CONTRAST_OPS +
    CV_KEYPOINT_OPS +
    CV_TEMPORAL_OPS +
    CV_POOLING_OPS
)

# ==================== MINIMAL OPERATION SET FOR FLAPPYBIRD ====================
# Carefully selected 27 operations for FlappyBird evolution
# Balanced across types with essential cross-type operations

FLAPPYBIRD_MINIMAL_OPS = [
    # SCALAR ARITHMETIC (8 ops) - Decision making
    AutoMLScalarAddOp,
    AutoMLScalarSubOp,
    AutoMLScalarMulOp,
    AutoMLScalarDivOp,
    AutoMLScalarMinOp,
    AutoMLScalarMaxOp,
    AutoMLScalarAbsOp,
    AutoMLScalarHeavisideOp,  # Critical for binary decisions
    
    # SCALAR TRIG (3 ops) - Non-linear transformations
    AutoMLScalarSinOp,        # Oscillation, periodic patterns
    AutoMLScalarCosOp,        # Oscillation, phase shift
    AutoMLScalarArctanOp,     # Angle computation, bounded output
    
    # VECTOR (5 ops) - Intermediate processing
    AutoMLVectorMeanOp,       # Vector → Scalar
    AutoMLVectorNormOp,       # Vector → Scalar
    AutoMLVectorAddOp,
    AutoMLVectorSubOp,
    AutoMLVectorDotOp,        # Vector × Vector → Scalar
    
    # MATRIX (6 ops) - Image manipulation
    AutoMLMatrixMeanOp,       # Matrix → Scalar
    AutoMLMatrixNormOp,       # Matrix → Scalar
    AutoMLMatrixAddOp,
    AutoMLMatrixSubOp,
    AutoMLScalarMatrixMulOp,  # Scalar × Matrix → Matrix
    AutoMLMatrixMeanAxis0Op,  # Matrix → Vector (column profile)
    
    # CV (5 ops) - Feature extraction
    CVGaussianBlurOp,         # Smoothing
    CVSobelXOp,               # Vertical edge detection
    CVSobelYOp,               # Horizontal edge detection
    CVMaxPoolOp,              # Downsampling (preserves edges)
    CVAvgPoolOp,              # Downsampling (smooths)
]

# ==================== MINIMAL SCALAR OPERATIONS FOR TESTING ====================
# Based on the 8 fundamental LGP operations: {+, -, *, /, cos, log, exp, conditional}

class ScalarConditionalOp(Operation):
    """Conditional operation: IF a < b THEN -a ELSE a
    
    This implements the classic LGP conditional operator that provides
    branching behavior without explicit control flow.
    """
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR, MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, a: float, b: float) -> float:
        return -a if a < b else a
    
    @property
    def name(self) -> str:
        return "scalar_conditional"
    
    @property
    def differentiable(self) -> bool:
        return False  # Has discontinuity at a=b


# Minimal 8 scalar operations for testing
# Reference: Brameier & Banzhaf's recommended minimal set
MINIMAL_SCALAR_OPS = [
    ScalarAddOp,              # +
    ScalarSubOp,              # -
    ScalarMulOp,              # *
    ScalarDivProtectedOp,     # / (protected)
    AutoMLScalarCosOp,        # cos
    AutoMLScalarLogOp,        # log (natural)
    AutoMLScalarExpOp,        # exp
    ScalarConditionalOp,      # IF a < b THEN -a ELSE a
]

# 12 operations optimized for feature_vector strategy
# Combines scalar ops with key vector aggregation ops
FEATURE_VECTOR_OPS = [
    # Scalar operations (8) - for combining individual features
    ScalarAddOp,              # +
    ScalarSubOp,              # -
    ScalarMulOp,              # *
    ScalarDivProtectedOp,     # / (protected)
    AutoMLScalarCosOp,        # cos (nonlinear)
    AutoMLScalarLogOp,        # log (nonlinear)
    AutoMLScalarExpOp,        # exp (nonlinear)
    ScalarConditionalOp,      # IF a < b THEN -a ELSE a
    # Vector operations (4) - for processing entire feature vector at once
    VectorDotProductOp,       # Weighted sum of all features (like a neural layer!)
    VectorMeanOp,             # Average all features
    VectorSumOp,              # Sum all features
    VectorNormOp,             # Magnitude/distance metric
]

# ==================== OPERATION REGISTRY ====================

# All scalar operations
SCALAR_OPS = [
    ScalarAddOp,
    ScalarSubOp,
    ScalarMulOp,
    ScalarDivProtectedOp,
    ScalarMaxOp,
    ScalarMinOp,
    ScalarAbsOp,
    ScalarNegOp,
    ScalarConditionalOp,
]

# All vector operations
VECTOR_OPS = [
    VectorAddOp,
    VectorSubOp,
    VectorMulOp,
    VectorDotProductOp,
    VectorMeanOp,
    VectorMaxOp,
    VectorMinOp,
    VectorSumOp,
    VectorNormOp,
]

# All matrix operations
MATRIX_OPS = [
    MatrixAddOp,
    MatrixSubOp,
    MatrixMulOp,
    MatrixMeanOp,
    MatrixMaxOp,
    MatrixMinOp,
    MatrixRowMeanOp,
    MatrixColMeanOp,
    MatrixRowMaxOp,
    MatrixColMaxOp,
    MatrixFlattenOp,
    MatrixTransposeOp,
]

# All cross-type operations
CROSS_TYPE_OPS = [
    ScalarVectorMulOp,
    ScalarVectorAddOp,
    ScalarMatrixMulOp,
    ScalarMatrixAddOp,
]

# All operations combined
ALL_OPS = SCALAR_OPS + VECTOR_OPS + MATRIX_OPS + CROSS_TYPE_OPS


def get_operations_by_output_type(output_type: MemoryType) -> List[type]:
    """Get all operation classes that produce a given output type"""
    return [op for op in ALL_OPS if op().output_type() == output_type]


def get_operations_by_input_types(input_types: List[MemoryType]) -> List[type]:
    """Get all operation classes that take specific input types"""
    return [op for op in ALL_OPS if op().input_types() == input_types]

if __name__ == "__main__":
    print("="*60)
    print("OPERATION TEST")
    print("="*60)
    
    # Test scalar operations
    print("\n--- Scalar Operations ---")
    add_op = ScalarAddOp()
    print(f"{add_op}: {add_op.execute(3.14, 2.71)}")
    
    div_op = ScalarDivProtectedOp()
    print(f"{div_op}: {div_op.execute(10.0, 0.0)} (protected)")
    
    # Test vector operations
    print("\n--- Vector Operations ---")
    v1 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    v2 = np.array([5, 4, 3, 2, 1], dtype=np.float32)
    
    vadd_op = VectorAddOp()
    print(f"{vadd_op}: {vadd_op.execute(v1, v2)}")
    
    vdot_op = VectorDotProductOp()
    print(f"{vdot_op}: {vdot_op.execute(v1, v2)}")
    
    vmean_op = VectorMeanOp()
    print(f"{vmean_op}: {vmean_op.execute(v1)}")
    
    # Test matrix operations
    print("\n--- Matrix Operations ---")
    m1 = np.random.rand(4, 4).astype(np.float32)
    
    mmean_op = MatrixMeanOp()
    print(f"{mmean_op}: {mmean_op.execute(m1):.4f}")
    
    mrow_op = MatrixRowMeanOp()
    print(f"{mrow_op}: {mrow_op.execute(m1)}")
    
    # Test cross-type operations
    print("\n--- Cross-Type Operations ---")
    sv_op = ScalarVectorMulOp()
    print(f"{sv_op}: {sv_op.execute(2.0, v1)}")
    
    # Test registry
    print("\n--- Operation Registry ---")
    print(f"Total operations: {len(ALL_OPS)}")
    print(f"Scalar ops: {len(SCALAR_OPS)}")
    print(f"Vector ops: {len(VECTOR_OPS)}")
    print(f"Matrix ops: {len(MATRIX_OPS)}")
    print(f"Cross-type ops: {len(CROSS_TYPE_OPS)}")
    
    scalar_producing = get_operations_by_output_type(MemoryType.SCALAR)
    print(f"\nOperations that produce scalars: {len(scalar_producing)}")
    for op_class in scalar_producing[:5]:
        print(f"  - {op_class().name}")
    
    print("\n✅ All operations working!")
