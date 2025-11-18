"""Fitness evaluation helpers for Linear Genetic Programming."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, List, Tuple

import numpy as np


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

            individual.program.execute(memory)

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
        # Patch extraction parameters (can be evolved in future)
        patch_size: int = 72,
        num_patches: int = 8,
        patch_strategy: str = "grid",  # "grid", "overlapping", "random", "strategic", "full", "full_image"
        # Note: For "full" strategy, patch_size determines the number of patches:
        #       pads image to square, then grid_size = padded_size / patch_size,
        #       resulting in grid_size² patches. num_patches is ignored for "full".
        # Note: For "full_image" strategy, returns a single 700x700 observation:
        #       crops to 700x576 (removes top/bottom 50px), pads to 700x700 (62px each side).
        #       patch_size and num_patches are ignored for "full_image".
        color_channel: int = 1,  # 0=R, 1=G, 2=B, or -1 for grayscale (mean)
        normalize: bool = True,  # Normalize to [0, 1]
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
        
        # Patch extraction configuration (evolvable parameters)
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.patch_strategy = patch_strategy
        self.color_channel = color_channel
        self.normalize = normalize

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

    def _resize_patch(self, patch: np.ndarray) -> np.ndarray:
        """Resize patch to configured patch_size."""
        target = self.patch_size
        if patch.shape == (target, target):
            return patch.astype(np.float32, copy=False)

        try:
            from scipy.ndimage import zoom

            zoom_factor = target / patch.shape[0], target / patch.shape[1]
            resized = zoom(patch, zoom_factor, order=1).astype(np.float32)
        except ImportError:
            # Nearest-neighbor fallback using numpy indexing
            src_h, src_w = patch.shape
            y_idx = np.linspace(0, max(src_h - 1, 0), target).astype(int)
            x_idx = np.linspace(0, max(src_w - 1, 0), target).astype(int)
            resized = patch[np.ix_(y_idx, x_idx)].astype(np.float32)

        return resized

    def _pad_to_square(self, image: np.ndarray) -> np.ndarray:
        """
        Pad image to square by adding zero columns/rows to the smaller dimension.
        
        Args:
            image: Single channel image array of shape (H, W)
            
        Returns:
            Square image array of shape (max(H, W), max(H, W))
        """
        h, w = image.shape
        max_dim = max(h, w)
        
        if h == w:
            return image.astype(np.float32, copy=False)
        
        # Create square array filled with zeros
        square_image = np.zeros((max_dim, max_dim), dtype=np.float32)
        
        if h < w:
            # Image is wider than tall: pad top and bottom
            pad_top = (max_dim - h) // 2
            square_image[pad_top:pad_top + h, :] = image
        else:
            # Image is taller than wide: pad left and right
            pad_left = (max_dim - w) // 2
            square_image[:, pad_left:pad_left + w] = image
        
        return square_image

    def _extract_full_coverage_patches(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Extract patches using square grid strategy:
        1. Pad image to square (using larger dimension)
        2. Calculate grid_size = padded_size / patch_size
        3. Extract grid_size × grid_size patches, each of size patch_size × patch_size
        4. No resizing - patches are used at their natural size
        1×1 → patches are 800×800 (1 patch)
2×2 → patches are 400×400 (4 patches)
4×4 → patches are 200×200 (16 patches)
5×5 → patches are 160×160 (25 patches)
8×8 → patches are 100×100 (64 patches)
10×10 → patches are 80×80 (100 patches)
16×16 → patches are 50×50 (256 patches)
20×20 → patches are 40×40 (400 patches)
25×25 → patches are 32×32 (625 patches)
32×32 → patches are 25×25 (1024 patches)
40×40 → patches are 20×20 (1600 patches)
50×50 → patches are 16×16 (2500 patches)
80×80 → patches are 10×10 (6400 patches)
100×100 → patches are 8×8 (10000 patches)
        
        Args:
            image: Single channel image array of shape (H, W)
            
        Returns:
            List of patch matrices, each of shape (patch_size, patch_size)
        """
        # Pad to square
        square_image = self._pad_to_square(image)
        padded_size = square_image.shape[0]  # Height and width are the same
        
        # Calculate grid size
        grid_size = padded_size // self.patch_size
        
        if grid_size == 0:
            # If patch_size is larger than image, return single patch (padded if needed)
            if padded_size < self.patch_size:
                patch = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
                patch[:padded_size, :padded_size] = square_image
                return [patch]
            else:
                return [square_image[:self.patch_size, :self.patch_size].copy()]
        
        # Extract patches from grid
        patches: List[np.ndarray] = []
        for i in range(grid_size):
            for j in range(grid_size):
                y0 = i * self.patch_size
                y1 = y0 + self.patch_size
                x0 = j * self.patch_size
                x1 = x0 + self.patch_size
                patch = square_image[y0:y1, x0:x1].copy()
                patches.append(patch)
        
        return patches

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
        Extract patches from image based on configured strategy.
        
        Args:
            image: Single channel image array of shape (H, W)
            
        Returns:
            List of patch matrices.
            - For "full" strategy: number of patches is determined by patch_size
              (grid_size² where grid_size = padded_size / patch_size).
            - For "full_image" strategy: returns exactly 1 patch of shape (700, 700).
            - For other strategies: returns exactly num_patches patches of shape (patch_size, patch_size).
        """
        h, w = image.shape
        patch_size = self.patch_size
        num_patches = self.num_patches
        
        patches = []
        
        if self.patch_strategy == "full":
            # Full coverage strategy: number of patches is determined by patch_size
            # Returns all patches generated (no truncation/padding to num_patches)
            return self._extract_full_coverage_patches(image)
        
        if self.patch_strategy == "full_image":
            # Full image strategy: crop to 700x576, pad to 700x700, return single observation
            # Returns exactly 1 patch (the full 700x700 image)
            return self._extract_full_image(image)

        if self.patch_strategy == "grid":
            # Regular grid: divide image into grid and select patches
            # Calculate grid dimensions
            grid_h = int(np.sqrt(num_patches * (h / w)))  # Approximate square grid
            grid_w = int(np.ceil(num_patches / grid_h))
            
            # Calculate step sizes
            step_h = max(1, (h - patch_size) // max(1, grid_h - 1)) if grid_h > 1 else 0
            step_w = max(1, (w - patch_size) // max(1, grid_w - 1)) if grid_w > 1 else 0
            
            # Extract patches from grid
            for i in range(grid_h):
                for j in range(grid_w):
                    if len(patches) >= num_patches:
                        break
                    y = min(i * step_h, h - patch_size)
                    x = min(j * step_w, w - patch_size)
                    patch = image[y:y+patch_size, x:x+patch_size]
                    patches.append(patch.copy())
                if len(patches) >= num_patches:
                    break
        
        elif self.patch_strategy == "overlapping":
            # Overlapping patches: slide window across image
            step = patch_size // 2  # 50% overlap
            for y in range(0, h - patch_size + 1, step):
                for x in range(0, w - patch_size + 1, step):
                    if len(patches) >= num_patches:
                        break
                    patch = image[y:y+patch_size, x:x+patch_size]
                    patches.append(patch.copy())
                if len(patches) >= num_patches:
                    break
        
        elif self.patch_strategy == "random":
            # Random patches: sample random locations
            for _ in range(num_patches):
                y = self.rng.integers(0, max(1, h - patch_size + 1))
                x = self.rng.integers(0, max(1, w - patch_size + 1))
                patch = image[y:y+patch_size, x:x+patch_size]
                patches.append(patch.copy())
        
        elif self.patch_strategy == "strategic":
            # Strategic: focus on center region where bird/pipes typically are
            # Center region: middle 60% of image
            center_y_start = int(h * 0.2)
            center_y_end = int(h * 0.8)
            center_x_start = int(w * 0.2)
            center_x_end = int(w * 0.8)
            
            center_h = center_y_end - center_y_start
            center_w = center_x_end - center_x_start
            
            # Grid within center region
            grid_h = int(np.sqrt(num_patches * (center_h / center_w)))
            grid_w = int(np.ceil(num_patches / grid_h))
            
            step_h = max(1, (center_h - patch_size) // max(1, grid_h - 1)) if grid_h > 1 else 0
            step_w = max(1, (center_w - patch_size) // max(1, grid_w - 1)) if grid_w > 1 else 0
            
            for i in range(grid_h):
                for j in range(grid_w):
                    if len(patches) >= num_patches:
                        break
                    y = center_y_start + min(i * step_h, center_h - patch_size)
                    x = center_x_start + min(j * step_w, center_w - patch_size)
                    patch = image[y:y+patch_size, x:x+patch_size]
                    patches.append(patch.copy())
                if len(patches) >= num_patches:
                    break
        
        else:
            raise ValueError(f"Unknown patch_strategy: {self.patch_strategy}")
        
        # Ensure we have exactly num_patches (pad with zeros if needed)
        while len(patches) < num_patches:
            patches.append(np.zeros((patch_size, patch_size), dtype=np.float32))
        
        return patches[:num_patches]

    def _process_observation(self, observation: np.ndarray) -> List[np.ndarray]:
        """
        Process the raw RGB observation into matrix patches for memory.
        
        This method extracts patches from the image based on configured parameters.
        The parameters (patch_size, num_patches, strategy, etc.) can be evolved
        in future implementations.
        
        Args:
            observation: RGB image array of shape (800, 576, 3) with values 0-255
            
        Returns:
            List of patch matrices, each of shape (patch_size, patch_size)
        """
        # Step 1: Extract color channel
        single_channel = self._extract_color_channel(observation)
        
        # Step 2: Extract patches based on strategy
        patches = self._extract_patches(single_channel)
        
        return patches

    def _evaluate_episode(self, individual: 'Individual', episode_idx: int) -> float:
        observation, _ = self.env.reset()
        observation = np.asarray(observation, dtype=np.float32)

        memory = individual.memory.copy()
        total_reward = 0.0

        for _ in range(self.max_steps):
            # Process image observation into matrices
            matrix_observations = self._process_observation(observation)
            memory.load_observation({'matrix': matrix_observations})

            individual.program.execute(memory)

            # Read action from output register
            action_value = memory.read_scalar(self.output_register)
            action = 1 if action_value >= 0.0 else 0

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