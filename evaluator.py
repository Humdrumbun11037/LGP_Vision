"""Fitness evaluation helpers for Linear Genetic Programming."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, List, Tuple

import numpy as np
from scipy.special import expit

import gymnasium as gym  # type: ignore[import]


from memory_system import MemoryType

from individual import Individual  


class FitnessEvaluator(ABC):
    """Base class for evaluating individuals.

    Subclasses implement `_evaluate_episode` to define how a single
    evaluation episode is scored. The public `evaluate` method averages
    across episodes and returns a scalar fitness.
    """

    def __init__(
        self,
        episodes: int = 1,
        rng: Optional[np.random.Generator] = None,
        output_registers: Optional[List[Tuple[MemoryType, int]]] = None,
    ) -> None:
        if episodes <= 0:
            raise ValueError("episodes must be positive")
        self.episodes = episodes
        self.rng = rng or np.random.default_rng()
        self.output_registers: List[Tuple[MemoryType, int]] = output_registers or []

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

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        super().__init__(episodes=5, rng=rng, output_registers=[(MemoryType.SCALAR, 0)])

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
        env_id: str = "CartPole-v1",
        episodes: int = 10,
        max_steps: int = 500,
        output_register: int = 7,
        render_mode: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if gym is None:
            raise ImportError("gymnasium is required for CartPoleEvaluator")
        super().__init__(
            episodes=episodes,
            rng=rng,
            output_registers=[(MemoryType.SCALAR, output_register)],
        )
        self.env = gym.make(env_id, render_mode=render_mode)
        self.max_steps = max_steps
        self.output_register = output_register
        self.env_id = env_id
        self.render_mode = render_mode

    def close(self) -> None:
        if hasattr(self, "env") and self.env is not None:
            self.env.close()

    def __del__(self):  # pragma: no cover
        self.close()

    def _evaluate_episode(self, individual: 'Individual', episode_idx: int) -> float:
        
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
        env_id: str = "FlappyBird-v0",
        episodes: int = 10,
        max_steps: int = 1000,
        output_register: int = 7,
        render_mode: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
        # Image processing parameters
        patch_strategy: str = "full_image",  # "full_image" or "quantized"
        # Note: For "full_image" strategy, returns a single 700x700 observation:
        #       crops to 700x576 (removes top/bottom 50px), pads to 700x700 (62px each side).
        # Note: For "quantized" strategy, pads image to 800x800 first, then downsamples using quantization_factor.
        color_channel: int = 1,  # 0=R, 1=G, 2=B, or -1 for grayscale (mean)
        normalize: bool = True,  # Normalize to [0, 1]
        quantization_factor: float = 0.5,  # Factor to downsample by (0.5 = half resolution). Only used for "quantized" strategy.
    ) -> None:
        if gym is None:
            raise ImportError("gymnasium is required for FlappyBirdEvaluator")
        try:
            import flappy_bird_env  # noqa
        except ImportError:
            raise ImportError("flappy-bird-env is required for FlappyBirdEvaluator")
        
        super().__init__(
            episodes=episodes,
            rng=rng,
            output_registers=[(MemoryType.SCALAR, output_register)],
        )
        self.env = gym.make(env_id, render_mode=render_mode)
        self.max_steps = max_steps
        self.output_register = output_register
        self.env_id = env_id
        self.render_mode = render_mode
        
        # Image processing configuration
        self.patch_strategy = patch_strategy
        self.color_channel = color_channel
        self.normalize = normalize
        self.quantization_factor = quantization_factor

    def close(self) -> None:
        if hasattr(self, "env") and self.env is not None:
            self.env.close()

    def __del__(self) -> None:  # pragma: no cover
        self.close()

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
            image: Single channel image array of shape (H, W)
            
        Returns:
            List containing a single matrix:
            - For "full_image" strategy: returns exactly 1 matrix of shape (700, 700).
            - For "quantized" strategy: pads image to 800x800 first, then downsamples and returns as single matrix.
        """
        if self.patch_strategy == "full_image":
            # Full image strategy: crop to 700x576, pad to 700x700, return single observation
            # Returns exactly 1 matrix (the full 700x700 image)
            return self._extract_full_image(image)

        elif self.patch_strategy == "quantized":
            # Quantized strategy: pad to 800x800 first, then downsample
            # This ensures consistent input size before quantization
            try:
                import cv2
            except ImportError:
                raise ImportError("OpenCV (cv2) is required for quantized strategy")
            
            h, w = image.shape
            
            # Step 1: Pad image to 800x800 by adding zeros on each side
            # Similar to full_image strategy but to 800x800 instead of 700x700
            target_size = 800
            padded = np.zeros((target_size, target_size), dtype=np.float32)
            
            # Center the image in the padded array
            pad_top = (target_size - h) // 2
            pad_left = (target_size - w) // 2
            padded[pad_top:pad_top + h, pad_left:pad_left + w] = image
            
            # Step 2: Downsample the 800x800 padded image using quantization_factor
            target_w = max(1, int(target_size * self.quantization_factor))
            target_h = max(1, int(target_size * self.quantization_factor))
            
            # Downsample using INTER_AREA interpolation (best for downsampling)
            # Convert to uint8 for cv2.resize, then back to float32
            if self.normalize:
                # Image is already normalized [0, 1], convert to [0, 255] for cv2
                padded_uint8 = (padded * 255.0).clip(0, 255).astype(np.uint8)
            else:
                # Image is [0, 255], just convert to uint8
                padded_uint8 = padded.clip(0, 255).astype(np.uint8)
            
            downsampled_uint8 = cv2.resize(
                padded_uint8,
                (target_w, target_h),
                interpolation=cv2.INTER_AREA
            )
            
            # Convert back to float32
            if self.normalize:
                downsampled = downsampled_uint8.astype(np.float32) / 255.0
            else:
                downsampled = downsampled_uint8.astype(np.float32)
            
            # Return as single matrix
            return [downsampled]
        
        else:
            raise ValueError(
                f"Unknown patch_strategy: {self.patch_strategy}. "
                f"Must be 'full_image' or 'quantized'"
            )

    def _process_observation(self, observation: np.ndarray) -> List[np.ndarray]:
        """
        Process the raw RGB observation into matrix observations for memory.
        
        This method processes the image based on configured strategy.
        
        Args:
            observation: RGB image array of shape (800, 576, 3) with values 0-255
            
        Returns:
            List containing a single matrix:
            - For "full_image": shape (700, 700)
            - For "quantized": shape depends on quantization_factor
        """
        # Step 1: Extract color channel
        single_channel = self._extract_color_channel(observation)
        
        # Step 2: Process image based on strategy (returns list with single matrix)
        matrix_observations = self._extract_patches(single_channel)
        
        return matrix_observations

    def _evaluate_episode(self, individual: 'Individual', episode_idx: int) -> float:
        observation, _ = self.env.reset()
        observation = np.asarray(observation, dtype=np.float32)

        memory = individual.memory.copy()
        total_reward = 0.0

        for _ in range(self.max_steps):
            # Process image observation into matrices
            matrix_observations = self._process_observation(observation)
            memory.load_observation({'matrix': matrix_observations})

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
        cartpole_eval = CartPoleEvaluator(episodes=2, rng=rng)
        # Ensure individual has properly sized memory for cartpole; reuse existing for demo.
        cartpole_fitness = cartpole_eval.evaluate(individual)
        print("CartPole fitness:", cartpole_fitness)