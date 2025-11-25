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

# try:
#     from vision_encoder import MobileNetV2FeatureExtractor
#     VISION_ENCODER_AVAILABLE = True
# except ImportError:
#     VISION_ENCODER_AVAILABLE = False
#     MobileNetV2FeatureExtractor = None


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
class FlappyBirdEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for FlappyBird evaluator."""
    env_id: str = "FlappyBird-v0"
    max_steps: int = 1000
    output_register: int = 0
    render_mode: Optional[str] = "rgb_array"
    patch_strategy: str = "quantized"  # "full_image", "quantized", or "feature_vector"
    color_channel: int = 1  # 0=R, 1=G, 2=B, or -1 for grayscale
    normalize: bool = True
    quantization_factor: float = 0.5
    feature_vector_size: int = 256  


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
        # Seed each episode differently to ensure variety
        # Use episode_idx to get different episodes even with same base seed
        episode_seed = None
        if self.config.rng_seed is not None:
            # Use config seed + episode index (deterministic and unique per worker)
            episode_seed = int((self.config.rng_seed + episode_idx * 100) % (2**31))
        
        if episode_seed is not None:
            observation, _ = self.env.reset(seed=episode_seed)
        else:
            observation, _ = self.env.reset()
        observation = np.asarray(observation, dtype=np.float32)

        memory = individual.memory.copy()
        total_reward = 0.0

        for _ in range(self.max_steps): # stateful registers 
            memory.load_observation({'scalar': observation.tolist()})
            individual.get_effective_program(self.output_registers).execute(memory)

            action_value = memory.read_scalar(self.output_register)
            action = 1 if action_value >= 0.0 else 0

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
        # Add flappy-bird-env submodule to path if it exists (for direct import without pip install)
        import sys
        from pathlib import Path
        _submodule_path = Path(__file__).parent.parent / "flappy-bird-env"
        if _submodule_path.exists() and str(_submodule_path) not in sys.path:
            sys.path.insert(0, str(_submodule_path))
        
        import flappy_bird_env  # noqa
       
        
        # Validate patch_strategy
        if config.patch_strategy not in ["full_image", "quantized", "feature_vector"]:
            raise ValueError(
                f"Unknown patch_strategy: {config.patch_strategy}. "
                f"Must be 'full_image', 'quantized', or 'feature_vector'"
            )
        
        # Validate feature_vector strategy requirements
        if config.patch_strategy == "feature_vector":
            if not VISION_ENCODER_AVAILABLE:
                raise ImportError(
                    "MobileNetV2FeatureExtractor is required for 'feature_vector' strategy. "
                    "Install PyTorch and torchvision: pip install torch torchvision"
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
            
            # Convert back to float32
            if self.normalize:
                quantized = quantized_uint8.astype(np.float32) / 255.0
            else:
                quantized = quantized_uint8.astype(np.float32)
            
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

    def _process_observation(self, observation: np.ndarray) -> Tuple[List[np.ndarray], str]:
        """
        Process the raw RGB observation into observations for memory.
        
        This method processes the image based on configured strategy.
        
        Args:
            observation: RGB image array of shape (800, 576, 3) with values 0-255
            
        Returns:
            Tuple of (observations_list, observation_type) where:
            - observation_type is 'matrix' for "full_image" and "quantized" strategies
            - observation_type is 'vector' for "feature_vector" strategy
            - observations_list contains the processed observations
        """
        if self.patch_strategy == "feature_vector":
            # For feature_vector strategy, pass raw RGB observation directly
            feature_vector = self._extract_features(observation)
            return [feature_vector], 'vector'
        else:
            # For full_image and quantized strategies, extract color channel first
            single_channel = self._extract_color_channel(observation)
            # Process image based on strategy (returns list with single matrix)
            matrix_observations = self._extract_patches(single_channel)
            return matrix_observations, 'matrix'

    def _evaluate_episode(self, individual: 'Individual', episode_idx: int) -> float:
        # Seed each episode differently to ensure variety
        # Use episode_idx to get different episodes even with same base seed
        episode_seed = None
        if self.config.rng_seed is not None:
            # Use config seed + episode index (deterministic and unique per worker)
            episode_seed = int((self.config.rng_seed + episode_idx * 100) % (2**31))
        
        if episode_seed is not None:
            observation, _ = self.env.reset(seed=episode_seed)
        else:
            observation, _ = self.env.reset()
        observation = np.asarray(observation, dtype=np.float32)

        memory = individual.memory.copy()
        total_reward = 0.0

        for _ in range(self.max_steps):
            # Process observation based on strategy
            processed_observations, obs_type = self._process_observation(observation)
            
            # Load observations into memory based on type
            if obs_type == 'vector':
                memory.load_observation({'vector': processed_observations})
            else:  # obs_type == 'matrix'
                memory.load_observation({'matrix': processed_observations})

            individual.get_effective_program(self.output_registers).execute(memory)


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