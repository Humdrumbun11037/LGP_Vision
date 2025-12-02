"""Fitness evaluation helpers for Linear Genetic Programming."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, List, Tuple, Union

import numpy as np
from scipy.special import expit

import gymnasium as gym  # type: ignore[import]


from memory_system import MemoryType

from individual import Individual

try:
    from vision_encoder import MobileNetV2FeatureExtractor
    VISION_ENCODER_AVAILABLE = True
except ImportError:
    VISION_ENCODER_AVAILABLE = False
    MobileNetV2FeatureExtractor = None


@dataclass
class BaseEvaluatorConfig:
    """Base configuration for all evaluators."""
    episodes: int = 1
    rng_seed: Optional[int] = None
    output_registers: Optional[List[Tuple[MemoryType, int]]] = None
    n_jobs: Optional[int] = None  # Parallelization control: None=auto, 1=sequential, >1=workers


@dataclass
class CartPoleEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for CartPole evaluator."""
    env_id: str = "CartPole-v1"
    max_steps: int = 500
    output_register: int = 7
    render_mode: Optional[str] = None


@dataclass
class AcrobotEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for Acrobot evaluator."""
    env_id: str = "Acrobot-v1"
    max_steps: int = 500
    output_register: int = 7
    render_mode: Optional[str] = None


@dataclass
class PendulumEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for Pendulum evaluator."""
    env_id: str = "Pendulum-v1"
    max_steps: int = 200
    output_register: int = 7
    render_mode: Optional[str] = None
    action_range: Tuple[float, float] = (-2.0, 2.0)  # Torque range for Pendulum


@dataclass
class FlappyBirdEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for FlappyBird evaluator."""
    env_id: str = "FlappyBird-v0"
    max_steps: int = 1000
    output_register: int = 0
    render_mode: Optional[str] = "rgb_array"
    patch_strategy: str = "quantized"  # "full_image", "quantized", "feature_vector", or "trinary"
    color_channel: int = 1  # 0=R, 1=G, 2=B, or -1 for grayscale
    normalize: bool = True
    quantization_factor: float = 0.5
    feature_vector_size: int = 256
    frame_stack_size: int = 1  # Number of frames to stack for quantized/trinary strategy (1 = no stacking)
    
    # Trinary strategy parameters
    trinary_crop_bottom: int = 100  # Pixels to crop from bottom
    trinary_resize_factor: float = 0.03  # Resize factor before stretching
    trinary_final_size: int = 21  # Final mask size (21x21)
    trinary_bird_h_min: int = 0  # Bird HSV hue minimum
    trinary_bird_h_max: int = 50  # Bird HSV hue maximum
    trinary_bird_s_min: int = 50  # Bird HSV saturation minimum
    trinary_bird_s_max: int = 255  # Bird HSV saturation maximum
    trinary_bird_v_min: int = 50  # Bird HSV value minimum
    trinary_bird_v_max: int = 255  # Bird HSV value maximum
    trinary_pipe_h_min: int = 35  # Pipe HSV hue minimum
    trinary_pipe_h_max: int = 45  # Pipe HSV hue maximum
    trinary_pipe_s_min: int = 40  # Pipe HSV saturation minimum
    trinary_pipe_s_max: int = 255  # Pipe HSV saturation maximum
    trinary_pipe_v_min: int = 40  # Pipe HSV value minimum
    trinary_pipe_v_max: int = 255  # Pipe HSV value maximum  


class FitnessEvaluator(ABC):
    """Base class for evaluating individuals.

    Subclasses implement `_evaluate_episode` to define how a single
    evaluation episode is scored. The public `evaluate` method averages
    across episodes and returns a scalar fitness.
    """

    def __init__(
        self,
        config: BaseEvaluatorConfig,
    ) -> None:
        if config.episodes <= 0:
            raise ValueError("episodes must be positive")
        self.config = config
        self.episodes = config.episodes
        self.rng = np.random.default_rng(config.rng_seed) if config.rng_seed is not None else np.random.default_rng()
        self.output_registers: List[Tuple[MemoryType, int]] = config.output_registers or []

    def evaluate(self, individual: 'Individual') -> float:
        rewards = [
            float(self._evaluate_episode(individual, episode_idx))
            for episode_idx in range(self.episodes)
        ]
        return float(np.mean(rewards)) if rewards else 0.0

    @abstractmethod
    def _evaluate_episode(self, individual: 'Individual', episode_idx: int) -> float:
        """Return the reward obtained by the individual in a single episode."""

class SymbolicRegressionEvaluator(FitnessEvaluator):
    """Simple evaluator for testing: fit z = x + y.

    - Two scalar observations (x, y) are provided in obs registers [-1], [-2].
    - The program's output is read from working scalar register 0.
    - Fitness is negative absolute error so that higher is better.
    """

    def __init__(self, config: Optional[BaseEvaluatorConfig] = None) -> None:
        if config is None:
            config = BaseEvaluatorConfig(
                episodes=5,
                output_registers=[(MemoryType.SCALAR, 0)]
            )
        super().__init__(config)

    def _evaluate_episode(self, individual: 'Individual', episode_idx: int) -> float:
        memory = individual.memory.copy()

        x, y = self.rng.uniform(-1.0, 1.0, size=2)
        memory.load_observation({'scalar': [x, y]})

        individual.program.execute(memory)

        predicted = memory.read_scalar(0)
        target = x + y
        error = abs(predicted - target)
        return -error

class CartPoleEvaluator(FitnessEvaluator):
    """Evaluate a policy on CartPole using scalar observations and output register.

    Assumptions:
        - Observation registers [-1], [-2], [-3], [-4] store cartpole state.
        - Working scalars include register `output_register` which encodes the action.
        - The program writes to the designated output register after execution.
    """

    def __init__(
        self,
        config: CartPoleEvaluatorConfig,
    ) -> None:
        if gym is None:
            raise ImportError("gymnasium is required for CartPoleEvaluator")
        
        self.config = config
        
        # CRITICAL: Set output_registers if not provided in config
        # This is needed for effective program calculation (intron removal)
        # If output_registers is None, derive it from output_register
        if config.output_registers is None:
            from memory_system import MemoryType
            config.output_registers = [(MemoryType.SCALAR, config.output_register)]
        
        super().__init__(config)
        self.env = gym.make(config.env_id, render_mode=config.render_mode)
        self.max_steps = config.max_steps
        self.output_register = config.output_register
        self.env_id = config.env_id
        self.render_mode = config.render_mode

    def close(self) -> None:
        if hasattr(self, "env") and self.env is not None:
            self.env.close()

    def __del__(self):  # pragma: no cover
        self.close()

    def _evaluate_episode(self, individual: 'Individual', episode_idx: int) -> float:
        # Use fixed seed for deterministic evaluation across all workers
        if self.config.rng_seed is not None:
            # Add episode_idx to the seed to ensure each episode is different
            # but still deterministic
            current_seed = self.config.rng_seed + episode_idx
            observation, _ = self.env.reset(seed=current_seed)
        else:
            observation, _ = self.env.reset()
        observation = np.asarray(observation, dtype=np.float32)

        memory = individual.memory.copy()
        total_reward = 0.0

        for _ in range(self.max_steps): # stateful registers 
            obs_matrix = np.tile(observation, (4,1))
            memory.load_observation({
            'scalar': observation.tolist(),           # 4 scalars
            'vector': [observation.tolist()],        # 1 vector of size 4
            'matrix': [obs_matrix]                    # 1 matrix of size 4x4
            })
        
            individual.program.execute(memory)

            action_value = memory.read_scalar(self.output_register)
            action = 1 if action_value >= 0.0 else 0

            observation, reward, terminated, truncated, _ = self.env.step(action)
            observation = np.asarray(observation, dtype=np.float32)
            total_reward += reward
            if terminated or truncated:
                break

        return float(total_reward)


class AcrobotEvaluator(FitnessEvaluator):
    """Evaluate a policy on Acrobot using scalar observations and output register.

    Acrobot is an underactuated double pendulum. The goal is to swing the lower link
    up to a vertical position.

    Assumptions:
        - Observation registers store 6 scalar values: [cos(theta1), sin(theta1), 
          cos(theta2), sin(theta2), thetaDot1, thetaDot2]
        - Working scalars include register `output_register` which encodes the action.
        - Action space is Discrete(3): 0 = -1 torque, 1 = 0 torque, 2 = +1 torque
        - The program writes to the designated output register after execution.
    """

    def __init__(
        self,
        config: AcrobotEvaluatorConfig,
    ) -> None:
        if gym is None:
            raise ImportError("gymnasium is required for AcrobotEvaluator")
        
        self.config = config
        
        # CRITICAL: Set output_registers if not provided in config
        if config.output_registers is None:
            from memory_system import MemoryType
            config.output_registers = [(MemoryType.SCALAR, config.output_register)]
        
        super().__init__(config)
        self.env = gym.make(config.env_id, render_mode=config.render_mode)
        self.max_steps = config.max_steps
        self.output_register = config.output_register
        self.env_id = config.env_id
        self.render_mode = config.render_mode

    def close(self) -> None:
        if hasattr(self, "env") and self.env is not None:
            self.env.close()

    def __del__(self) -> None:  # pragma: no cover
        self.close()

    def _evaluate_episode(self, individual: 'Individual', episode_idx: int) -> float:
        # Use fixed seed for deterministic evaluation across all workers
        if self.config.rng_seed is not None:
            # Add episode_idx to the seed to ensure each episode is different
            # but still deterministic
            current_seed = self.config.rng_seed + episode_idx
            observation, _ = self.env.reset(seed=current_seed)
        else:
            observation, _ = self.env.reset()
        observation = np.asarray(observation, dtype=np.float32)

        memory = individual.memory.copy()
        total_reward = 0.0

        for _ in range(self.max_steps):
            # Acrobot has 6 observation dimensions
            obs_matrix = np.tile(observation, (6, 1))  # Shape: (6, 6)
            memory.load_observation({
                'scalar': observation.tolist(),           # 6 scalars
                'vector': [observation.tolist()],        # 1 vector of size 6
                'matrix': [obs_matrix]                    # 1 matrix of size 6x6
            })
            
            individual.program.execute(memory)

            # Acrobot action space is Discrete(3): 0, 1, 2
            # Map scalar output to discrete action: negative -> 0, zero -> 1, positive -> 2
            action_value = memory.read_scalar(self.output_register)
            if action_value < -0.33:
                action = 0  # Apply -1 torque
            elif action_value > 0.33:
                action = 2  # Apply +1 torque
            else:
                action = 1  # Apply 0 torque

            observation, reward, terminated, truncated, _ = self.env.step(action)
            observation = np.asarray(observation, dtype=np.float32)
            total_reward += reward
            if terminated or truncated:
                break

        return float(total_reward)


class PendulumEvaluator(FitnessEvaluator):
    """Evaluate a policy on Pendulum using scalar observations and output register.

    Pendulum is a classic control problem where the goal is to swing up and balance
    a pendulum in the upright position.

    Assumptions:
        - Observation registers store 3 scalar values: [cos(theta), sin(theta), thetaDot]
        - Working scalars include register `output_register` which encodes the action.
        - Action space is continuous: torque in range [-2.0, 2.0]
        - The program writes to the designated output register after execution.
    """

    def __init__(
        self,
        config: PendulumEvaluatorConfig,
    ) -> None:
        if gym is None:
            raise ImportError("gymnasium is required for PendulumEvaluator")
        
        self.config = config
        
        # CRITICAL: Set output_registers if not provided in config
        if config.output_registers is None:
            from memory_system import MemoryType
            config.output_registers = [(MemoryType.SCALAR, config.output_register)]
        
        super().__init__(config)
        self.env = gym.make(config.env_id, render_mode=config.render_mode)
        self.max_steps = config.max_steps
        self.output_register = config.output_register
        self.env_id = config.env_id
        self.render_mode = config.render_mode
        self.action_range = config.action_range

    def close(self) -> None:
        if hasattr(self, "env") and self.env is not None:
            self.env.close()

    def __del__(self) -> None:  # pragma: no cover
        self.close()

    def _evaluate_episode(self, individual: 'Individual', episode_idx: int) -> float:
        # Use fixed seed for deterministic evaluation across all workers
        if self.config.rng_seed is not None:
            # Add episode_idx to the seed to ensure each episode is different
            # but still deterministic
            current_seed = self.config.rng_seed + episode_idx
            observation, _ = self.env.reset(seed=current_seed)
        else:
            observation, _ = self.env.reset()
        observation = np.asarray(observation, dtype=np.float32)

        memory = individual.memory.copy()
        total_reward = 0.0

        for _ in range(self.max_steps):
            # Pendulum has 3 observation dimensions
            obs_matrix = np.tile(observation, (3, 1))  # Shape: (3, 3)
            memory.load_observation({
                'scalar': observation.tolist(),           # 3 scalars
                'vector': [observation.tolist()],        # 1 vector of size 3
                'matrix': [obs_matrix]                    # 1 matrix of size 3x3
            })
            
            individual.program.execute(memory)

            # Pendulum action space is continuous: clip to [-2.0, 2.0] range
            action_value = memory.read_scalar(self.output_register)
            action = np.clip(action_value, self.action_range[0], self.action_range[1])
            # Pendulum expects action as array
            action = np.array([action], dtype=np.float32)

            observation, reward, terminated, truncated, _ = self.env.step(action)
            observation = np.asarray(observation, dtype=np.float32)
            total_reward += reward
            if terminated or truncated:
                break

        return float(total_reward)


class FlappyBirdEvaluator(FitnessEvaluator):
    """Evaluate a policy on FlappyBird using image observations and output register.
    
    Assumptions:
        - Observation registers store image patches/matrices from the game screen.
        - Working scalars include register `output_register` which encodes the action.
        - The program writes to the designated output register after execution.
        - Action 0 = NOOP, Action 1 = flap wings
    """

    def __init__(
        self,
        config: FlappyBirdEvaluatorConfig,
    ) -> None:
        import flappy_bird_env  # noqa
       
        
        # Validate patch_strategy
        if config.patch_strategy not in ["full_image", "quantized", "feature_vector", "trinary"]:
            raise ValueError(
                f"Unknown patch_strategy: {config.patch_strategy}. "
                f"Must be 'full_image', 'quantized', 'feature_vector', or 'trinary'"
            )
        
        # Validate frame_stack_size
        if config.frame_stack_size < 1:
            raise ValueError(
                f"frame_stack_size must be >= 1, got {config.frame_stack_size}"
            )
        
        # Validate feature_vector strategy requirements
        if config.patch_strategy == "feature_vector":
            if not VISION_ENCODER_AVAILABLE:
                raise ImportError(
                    "MobileNetV2FeatureExtractor is required for 'feature_vector' strategy. "
                    "Install PyTorch and torchvision: pip install torch torchvision"
                )
        
        # Validate trinary strategy requirements
        if config.patch_strategy == "trinary":
            try:
                import cv2  # noqa
            except ImportError:
                raise ImportError(
                    "OpenCV (cv2) is required for 'trinary' strategy. "
                    "Install OpenCV: pip install opencv-python"
                )
        
        self.config = config
        
        # CRITICAL: Set output_registers if not provided in config
        # This is needed for effective program calculation (intron removal)
        # If output_registers is None, derive it from output_register
        if config.output_registers is None:
            from memory_system import MemoryType
            config.output_registers = [(MemoryType.SCALAR, config.output_register)]
        
        super().__init__(config)
        self.env = gym.make(config.env_id, render_mode=config.render_mode)
        self.max_steps = config.max_steps
        self.output_register = config.output_register
        self.env_id = config.env_id
        self.render_mode = config.render_mode
        
        # Image processing configuration
        self.patch_strategy = config.patch_strategy
        self.color_channel = config.color_channel
        self.normalize = config.normalize
        self.quantization_factor = config.quantization_factor
        self.feature_vector_size = config.feature_vector_size
        self.frame_stack_size = config.frame_stack_size
        
        # Trinary strategy configuration
        self.trinary_crop_bottom = config.trinary_crop_bottom
        self.trinary_resize_factor = config.trinary_resize_factor
        self.trinary_final_size = config.trinary_final_size
        self.trinary_bird_h_min = config.trinary_bird_h_min
        self.trinary_bird_h_max = config.trinary_bird_h_max
        self.trinary_bird_s_min = config.trinary_bird_s_min
        self.trinary_bird_s_max = config.trinary_bird_s_max
        self.trinary_bird_v_min = config.trinary_bird_v_min
        self.trinary_bird_v_max = config.trinary_bird_v_max
        self.trinary_pipe_h_min = config.trinary_pipe_h_min
        self.trinary_pipe_h_max = config.trinary_pipe_h_max
        self.trinary_pipe_s_min = config.trinary_pipe_s_min
        self.trinary_pipe_s_max = config.trinary_pipe_s_max
        self.trinary_pipe_v_min = config.trinary_pipe_v_min
        self.trinary_pipe_v_max = config.trinary_pipe_v_max
        
        # Initialize frame buffer for frame stacking (quantized and trinary strategies)
        self.frame_buffer: Optional[List[np.ndarray]] = None
        
        # Initialize feature extractor if using feature_vector strategy
        if config.patch_strategy == "feature_vector":
            self.feature_extractor = MobileNetV2FeatureExtractor(
                feature_size=config.feature_vector_size
            )
        else:
            self.feature_extractor = None

    def close(self) -> None:
        if hasattr(self, "env") and self.env is not None:
            self.env.close()

    def __del__(self) -> None:  # pragma: no cover
        self.close()

    def _extract_features(self, observation: np.ndarray) -> np.ndarray:
        """
        Extract feature vector from RGB observation using MobileNetV2.
        
        Args:
            observation: RGB image array of shape (H, W, 3) with values 0-255
            
        Returns:
            1D numpy array of shape (feature_vector_size,) with dtype float32
        """
        if self.feature_extractor is None:
            raise RuntimeError(
                "Feature extractor not initialized. "
                "This method should only be called when patch_strategy='feature_vector'"
            )
        
        return self.feature_extractor.extract_features(observation)
    
    def _extract_color_channel(self, observation: np.ndarray) -> np.ndarray:
        """
        Extract and normalize a color channel from RGB observation.
        
        Args:
            observation: RGB image array of shape (H, W, 3) with values 0-255
            
        Returns:
            Single channel image of shape (H, W) with values 0.0-1.0 (if normalize=True)
        """
        if self.color_channel == -1:
            # Grayscale: average all channels
            channel = np.mean(observation, axis=2)
        else:
            # Single channel (0=R, 1=G, 2=B)
            channel = observation[:, :, self.color_channel]
        
        channel = channel.astype(np.float32)
        
        if self.normalize:
            channel = channel / 255.0
        
        return channel

    def _extract_full_image(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Extract a single full image observation:
        1. Crop image to 700x576 (remove top 50 and bottom 50 pixels)
        2. Pad each side with 62 columns of zeros to make 700x700
        3. Return single 700x700 matrix as one observation
        
        Args:
            image: Single channel image array of shape (H, W), typically (800, 576)
            
        Returns:
            List containing a single matrix of shape (700, 700)
        """
        h, w = image.shape
        
        # Crop: remove top 50 and bottom 50 pixels
        crop_top = 50
        crop_bottom = 50
        cropped = image[crop_top:h-crop_bottom, :]  # Shape: (700, 576)
        
        # Pad: add 62 columns of zeros on each side to make 700x700
        pad_left = 62
        pad_right = 62
        padded = np.zeros((700, 700), dtype=np.float32)
        padded[:, pad_left:pad_left + 576] = cropped
        
        return [padded]

    def _extract_patches(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Process image based on configured strategy.
        
        Args:
            image: Single channel image array of shape (H, W) for "full_image" and "quantized" strategies.
                   For "feature_vector" strategy, this should be the raw RGB observation.
            
        Returns:
            List containing:
            - For "full_image" strategy: exactly 1 matrix of shape (700, 700).
            - For "quantized" strategy: exactly 1 matrix (downsampled).
            - For "feature_vector" strategy: exactly 1 feature vector (1D array).
        """
        if self.patch_strategy == "full_image":
            # Full image strategy: crop to 700x576, pad to 700x700, return single observation
            # Returns exactly 1 matrix (the full 700x700 image)
            return self._extract_full_image(image)

        elif self.patch_strategy == "quantized":
            # Quantized strategy:
            # 1. Trim bottom 50 rows (green channel only) -> 750 rows by 576 pixels
            # 2. Quantize by quantization_factor
            # 3. Stretch horizontally to create a square matrix
            try:
                import cv2
            except ImportError:
                raise ImportError("OpenCV (cv2) is required for quantized strategy")
            
            h, w = image.shape
            
            # Step 1: Trim bottom 50 rows
            # This gives us 750 rows by 576 pixels
            crop_bottom = 50
            trimmed = image[:h-crop_bottom, :]  # Shape: (750, 576)
            
            # Step 2: Quantize by quantization_factor
            quantized_h = max(1, int(trimmed.shape[0] * self.quantization_factor))
            quantized_w = max(1, int(trimmed.shape[1] * self.quantization_factor))
            
            # Convert to uint8 for cv2.resize, then back to float32
            if self.normalize:
                # Image is already normalized [0, 1], convert to [0, 255] for cv2
                trimmed_uint8 = (trimmed * 255.0).clip(0, 255).astype(np.uint8)
            else:
                # Image is [0, 255], just convert to uint8
                trimmed_uint8 = trimmed.clip(0, 255).astype(np.uint8)
            
            quantized_uint8 = cv2.resize(
                trimmed_uint8,
                (quantized_w, quantized_h),
                interpolation=cv2.INTER_AREA
            )
            
          
            # Step 3: Stretch horizontally to create a square matrix
            # The final height is quantized_h, so we stretch width to match
            final_size = quantized_h  # Square matrix: height = width
            squared_uint8 = cv2.resize(
                quantized_uint8,
                (final_size, final_size),
                interpolation=cv2.INTER_LINEAR  # Use linear interpolation for stretching
            )
            
            # Convert back to float32
            if self.normalize:
                squared = squared_uint8.astype(np.float32) / 255.0
            else:
                squared = squared_uint8.astype(np.float32)
            
            # Return as single matrix
            return [squared]
        
        elif self.patch_strategy == "feature_vector":
            # Feature vector strategy: extract features using MobileNetV2
            # Note: For this strategy, 'image' parameter should actually be the raw RGB observation
            # This is handled in _process_observation
            feature_vector = self._extract_features(image)
            return [feature_vector]
        
        else:
            raise ValueError(
                f"Unknown patch_strategy: {self.patch_strategy}. "
                f"Must be 'full_image', 'quantized', or 'feature_vector'"
            )

    def _extract_trinary_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Extract trinary semantic mask from RGB image.
        
        Creates a semantic mask with values:
        - -1.0 for bird pixels (HSV color range + white pixels)
        - 0.0 for background pixels
        - 1.0 for pipe pixels (HSV color range)
        
        Args:
            image: RGB image (H, W, 3) with values 0-255
            
        Returns:
            2D float32 array (final_size, final_size) with values -1, 0, or 1
        """
        import cv2
        
        # 1. Crop bottom pixels to remove ground
        h, w = image.shape[:2]
        cropped = image[:h-self.trinary_crop_bottom, :]
        
        # 2. Resize by factor (nearest neighbor to preserve discrete values)
        if self.trinary_resize_factor != 1.0:
            new_h = max(1, int(cropped.shape[0] * self.trinary_resize_factor))
            new_w = max(1, int(cropped.shape[1] * self.trinary_resize_factor))
            cropped = cv2.resize(cropped.astype(np.uint8), (new_w, new_h), 
                                interpolation=cv2.INTER_NEAREST)
        
        # 3. Stretch to final_size (nearest neighbor to preserve discrete values)
        cropped = cv2.resize(cropped.astype(np.uint8), 
                            (self.trinary_final_size, self.trinary_final_size),
                            interpolation=cv2.INTER_NEAREST)
        
        # 4. Convert RGB to HSV for better color segmentation
        hsv = cv2.cvtColor(cropped.astype(np.uint8), cv2.COLOR_RGB2HSV)
        
        # 5. Initialize mask as background (0)
        mask = np.zeros((self.trinary_final_size, self.trinary_final_size), dtype=np.float32)
        
        # 6. Bird detection (HSV range + white pixels)
        bird_lower = np.array([self.trinary_bird_h_min, self.trinary_bird_s_min, self.trinary_bird_v_min])
        bird_upper = np.array([self.trinary_bird_h_max, self.trinary_bird_s_max, self.trinary_bird_v_max])
        bird_mask = cv2.inRange(hsv, bird_lower, bird_upper)
        
        # Also detect white pixels (low saturation, high value) for bird's eye/white features
        # White in HSV: S < 30, V > 200 (hardcoded as specified)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
        
        # Combine bird color mask and white mask
        bird_mask = cv2.bitwise_or(bird_mask, white_mask)
        mask[bird_mask > 0] = -1.0
        
        # 7. Pipe detection (HSV range)
        pipe_lower = np.array([self.trinary_pipe_h_min, self.trinary_pipe_s_min, self.trinary_pipe_v_min])
        pipe_upper = np.array([self.trinary_pipe_h_max, self.trinary_pipe_s_max, self.trinary_pipe_v_max])
        pipe_mask = cv2.inRange(hsv, pipe_lower, pipe_upper)
        mask[pipe_mask > 0] = 1.0
        
        # Background is already 0 (default)
        return mask

    def _process_observation(self, observation: np.ndarray) -> Union[Tuple[List[np.ndarray], str], dict]:
        """
        Process the raw RGB observation into observations for memory.
        
        This method processes the image based on configured strategy.
        
        Args:
            observation: RGB image array of shape (800, 576, 3) with values 0-255
            
        Returns:
            For "full_image", "quantized", and "trinary" strategies:
                Tuple of (observations_list, 'matrix')
            For "feature_vector" strategy:
                Dict with both 'vector' and 'matrix' keys, where:
                - 'vector': [feature_vector] - the raw 1D feature vector
                - 'matrix': [tiled_matrix] - feature vector tiled as columns (N×N matrix)
        """
        if self.patch_strategy == "feature_vector":
            # Extract feature vector from RGB observation
            feature_vector = self._extract_features(observation)
            
            # Create tiled matrix: each column is the feature vector
            # Shape: (feature_size, feature_size) e.g., (64, 64) or (256, 256)
            feature_matrix = np.tile(
                feature_vector.reshape(-1, 1),  # Column vector
                (1, len(feature_vector))         # Tile N times horizontally
            ).astype(np.float32)
            
            # Return BOTH vector and matrix observations
            return {
                'vector': [feature_vector],
                'matrix': [feature_matrix]
            }
        elif self.patch_strategy == "trinary":
            # Extract trinary mask from RGB observation (no color channel extraction needed)
            current_frame_matrix = self._extract_trinary_mask(observation)
            
            # Handle frame stacking for trinary strategy (same logic as quantized)
            if self.frame_stack_size > 1:
                # Initialize or update frame buffer
                if self.frame_buffer is None:
                    # First frame: copy it N times
                    self.frame_buffer = [current_frame_matrix.copy() for _ in range(self.frame_stack_size)]
                else:
                    # Sliding window: remove oldest, add newest
                    self.frame_buffer.pop(0)
                    self.frame_buffer.append(current_frame_matrix.copy())
                
                # Return list of stacked frames (deep copy to avoid modification)
                stacked_frames = [frame.copy() for frame in self.frame_buffer]
                return stacked_frames, 'matrix'
            else:
                # No frame stacking: return single matrix (backward compatible)
                return [current_frame_matrix], 'matrix'
        else:
            # For full_image and quantized strategies, extract color channel first
            single_channel = self._extract_color_channel(observation)
            # Process image based on strategy (returns list with single matrix)
            current_frame_matrix = self._extract_patches(single_channel)[0]  # Get the single matrix
            
            # Handle frame stacking for quantized strategy
            if self.patch_strategy == "quantized" and self.frame_stack_size > 1:
                # Initialize or update frame buffer
                if self.frame_buffer is None:
                    # First frame: copy it N times
                    self.frame_buffer = [current_frame_matrix.copy() for _ in range(self.frame_stack_size)]
                else:
                    # Sliding window: remove oldest, add newest
                    self.frame_buffer.pop(0)
                    self.frame_buffer.append(current_frame_matrix.copy())
                
                # Return list of stacked frames (deep copy to avoid modification)
                stacked_frames = [frame.copy() for frame in self.frame_buffer]
                return stacked_frames, 'matrix'
            else:
                # No frame stacking: return single matrix (backward compatible)
                return [current_frame_matrix], 'matrix'

    def _evaluate_episode(self, individual: 'Individual', episode_idx: int) -> float:
        # Use fixed seed for deterministic evaluation across all workers
        if self.config.rng_seed is not None:
            # Add episode_idx to the seed to ensure each episode is different
            # but still deterministic
            current_seed = self.config.rng_seed + episode_idx
            observation, _ = self.env.reset(seed=current_seed)
        else:
            observation, _ = self.env.reset()
        observation = np.asarray(observation, dtype=np.float32)

        # Reset frame buffer for new episode
        self.frame_buffer = None

        memory = individual.memory.copy()
        total_reward = 0.0
        
        # Validate frame stacking requirements
        if self.frame_stack_size > 1 and self.patch_strategy in ["quantized", "trinary"]:
            if memory.n_obs_matrix < self.frame_stack_size:
                raise ValueError(
                    f"Frame stacking requires n_obs_matrix >= frame_stack_size. "
                    f"Got n_obs_matrix={memory.n_obs_matrix}, "
                    f"frame_stack_size={self.frame_stack_size}"
                )

        for _ in range(self.max_steps):
            # Process observation based on strategy
            processed = self._process_observation(observation)
            
            # Load observations into memory based on return type
            if isinstance(processed, dict):
                # feature_vector strategy returns dict with both 'vector' and 'matrix'
                memory.load_observation(processed)
            else:
                # full_image/quantized strategies return (observations_list, obs_type) tuple
                processed_observations, obs_type = processed
                if obs_type == 'vector':
                    memory.load_observation({'vector': processed_observations})
                else:  # obs_type == 'matrix'
                    memory.load_observation({'matrix': processed_observations})

            # individual.get_effective_program(self.output_registers).execute(memory)
            individual.program.execute(memory)


            # Read action from output register
            action_value = memory.read_scalar(self.output_register)
            
            normalized = expit(action_value)  # expit is the sigmoid function   
            
            action = 1 if normalized >= 0.5 else 0

            observation, reward, terminated, truncated, _ = self.env.step(action)
            observation = np.asarray(observation, dtype=np.float32)
            total_reward += reward
            
            if terminated or truncated:
                break

        return float(total_reward)


@dataclass
class FlappyBirdSimpleEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for FlappyBird evaluator with 12-feature vector observations."""
    env_id: str = "FlappyBird-v0"
    max_steps: int = 1000
    output_register: int = 0
    render_mode: Optional[str] = None
    use_lidar: bool = False  # Use 12-feature vector instead of LIDAR


class FlappyBirdSimpleEvaluator(FitnessEvaluator):
    """
    Evaluate a policy on FlappyBird using 12-feature vector scalar observations.
    
    Uses flappy-bird-gymnasium with use_lidar=False to get 12 scalar features:
    - last pipe's horizontal position
    - last top pipe's vertical position
    - last bottom pipe's vertical position
    - next pipe's horizontal position
    - next top pipe's vertical position
    - next bottom pipe's vertical position
    - next next pipe's horizontal position
    - next next top pipe's vertical position
    - next next bottom pipe's vertical position
    - player's vertical position
    - player's vertical velocity
    - player's rotation
    
    Assumptions:
        - Observation registers store the 12 scalar features
        - Working scalars include register `output_register` which encodes the action
        - The program writes to the designated output register after execution
    """
    
    def __init__(
        self,
        config: FlappyBirdSimpleEvaluatorConfig,
    ) -> None:
        try:
            import flappy_bird_gymnasium
        except ImportError:
            raise ImportError(
                "flappy-bird-gymnasium is required for FlappyBirdSimpleEvaluator. "
                "Install it with: pip install flappy-bird-gymnasium"
            )
        
        if gym is None:
            raise ImportError("gymnasium is required for FlappyBirdSimpleEvaluator")
        
        self.config = config
        
        # CRITICAL: Set output_registers if not provided in config
        if config.output_registers is None:
            from memory_system import MemoryType
            config.output_registers = [(MemoryType.SCALAR, config.output_register)]
        
        super().__init__(config)
        
        # Create environment with use_lidar=False for 12-feature vector
        self.env = gym.make(
            config.env_id,
            render_mode=config.render_mode,
            use_lidar=config.use_lidar
        )
        self.max_steps = config.max_steps
        self.output_register = config.output_register
        self.env_id = config.env_id
        self.render_mode = config.render_mode
        self.use_lidar = config.use_lidar
    
    def close(self) -> None:
        if hasattr(self, "env") and self.env is not None:
            self.env.close()
    
    def __del__(self) -> None:  # pragma: no cover
        self.close()
    
    def _evaluate_episode(self, individual: 'Individual', episode_idx: int) -> float:
        # Use fixed seed for deterministic evaluation across all workers
        if self.config.rng_seed is not None:
            # Add episode_idx to the seed to ensure each episode is different
            # but still deterministic
            current_seed = self.config.rng_seed + episode_idx
            observation, _ = self.env.reset(seed=current_seed)
        else:
            observation, _ = self.env.reset()
        observation = np.asarray(observation, dtype=np.float32)
        
        memory = individual.memory.copy()
        total_reward = 0.0
        
        # The observation is a 12-element feature vector
        obs_size = len(observation)
        
        for _ in range(self.max_steps):
            # Load observation as scalars, vector, and matrix (tiled)
            # Similar to CartPole but with 12 features instead of 4
            obs_matrix = np.tile(observation, (obs_size, 1))
            
            memory.load_observation({
                'scalar': observation.tolist(),          # 12 scalars
                'vector': [observation.tolist()],       # 1 vector of size 12
                'matrix': [obs_matrix]                   # 1 matrix of size 12x12
            })
            
            individual.program.execute(memory)
            
            # Read action from output register
            action_value = memory.read_scalar(self.output_register)
            
            # Use sigmoid to normalize action value, then threshold at 0.5
            from scipy.special import expit
            normalized = expit(action_value)  # expit is the sigmoid function
            action = 1 if normalized >= 0.5 else 0
            
            observation, reward, terminated, truncated, _ = self.env.step(action)
            observation = np.asarray(observation, dtype=np.float32)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        return float(total_reward)


if __name__ == "__main__":
    from memory_system import MemoryConfig, MemoryBank
    from instruction_set import InstructionSet
    from operation import ALL_OPS, SCALAR_OPS
    from individual import Individual  # type: ignore[import]

    rng = np.random.default_rng(0)

    memory_cfg = MemoryConfig(
        n_scalar=8,
        n_vector=0,
        n_matrix=0,
        n_obs_scalar=4,
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
    individual = Individual.random(instr_set, memory_cfg, program_length=6, rng=rng)

    # evaluator = SymbolicRegressionEvaluator(rng)
    # fitness = evaluator.evaluate(individual)
    # print("Symbolic regression fitness:", fitness)

    if gym is not None:
        cartpole_config = CartPoleEvaluatorConfig(episodes=2, rng_seed=0)
        cartpole_eval = CartPoleEvaluator(config=cartpole_config)
        # Ensure individual has properly sized memory for cartpole; reuse existing for demo.
        cartpole_fitness = cartpole_eval.evaluate(individual)
        print("CartPole fitness:", cartpole_fitness)