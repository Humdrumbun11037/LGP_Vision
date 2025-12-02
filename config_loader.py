"""Configuration loader for YAML-based configuration."""

import yaml
from pathlib import Path
from typing import Optional, Dict, Any, TYPE_CHECKING

from memory_system import MemoryConfig
from evaluator import FlappyBirdEvaluatorConfig
from population import PopulationConfig
from evolution_engine import EvolutionConfig

if TYPE_CHECKING:
    from experiment_manager import ExperimentManager


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing all configuration sections
        
    Raises:
        FileNotFoundError: If the configuration file does not exist
        yaml.YAMLError: If the YAML file is malformed
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}")
    
    if config is None:
        raise ValueError(f"Configuration file {config_path} is empty")
    
    return config


def create_memory_config(config: Dict[str, Any]) -> MemoryConfig:
    """Create MemoryConfig from YAML config.
    
    Args:
        config: Dictionary containing configuration sections
        
    Returns:
        MemoryConfig object
        
    Raises:
        ValueError: If required fields are missing or invalid
    """
    mem_cfg = config.get('memory', {})
    
    # Validate required fields
    required_fields = ['n_scalar', 'n_vector', 'n_matrix', 'n_obs_scalar', 
                       'n_obs_vector', 'n_obs_matrix', 'vector_size', 'matrix_shape']
    missing = [field for field in required_fields if field not in mem_cfg]
    if missing:
        raise ValueError(f"Missing required memory configuration fields: {missing}")
    
    # Convert matrix_shape from list to tuple
    matrix_shape = mem_cfg.get('matrix_shape', [37, 37])
    if isinstance(matrix_shape, list):
        if len(matrix_shape) != 2:
            raise ValueError(f"matrix_shape must be a 2-element list, got: {matrix_shape}")
        matrix_shape = tuple(matrix_shape)
    elif not isinstance(matrix_shape, tuple):
        raise ValueError(f"matrix_shape must be a list or tuple, got: {type(matrix_shape)}")
    
    return MemoryConfig(
        n_scalar=mem_cfg['n_scalar'],
        n_vector=mem_cfg['n_vector'],
        n_matrix=mem_cfg['n_matrix'],
        n_obs_scalar=mem_cfg['n_obs_scalar'],
        n_obs_vector=mem_cfg['n_obs_vector'],
        n_obs_matrix=mem_cfg['n_obs_matrix'],
        vector_size=mem_cfg['vector_size'],
        matrix_shape=matrix_shape,
    )


def create_evaluator_config(config: Dict[str, Any]) -> FlappyBirdEvaluatorConfig:
    """Create FlappyBirdEvaluatorConfig from YAML config.
    
    Args:
        config: Dictionary containing configuration sections
        
    Returns:
        FlappyBirdEvaluatorConfig object
        
    Raises:
        ValueError: If required fields are missing or invalid
    """
    eval_cfg = config.get('evaluator', {})
    
    # Handle headless mode parameter (takes precedence over render_mode if render_mode is null)
    headless = eval_cfg.get('headless', True)  # Default to headless for speed
    
    # Handle null render_mode (convert to None)
    render_mode = eval_cfg.get('render_mode')
    if render_mode == 'null' or render_mode is None:
        # If render_mode is not specified, use headless flag to determine it
        render_mode = "rgb_array" if headless else "human"
    # If render_mode is explicitly set, use it (headless flag is ignored)
    
    # Handle null n_jobs (convert to None)
    n_jobs = eval_cfg.get('n_jobs')
    if n_jobs == 'null' or n_jobs is None:
        n_jobs = None
    elif isinstance(n_jobs, str) and n_jobs.lower() == 'null':
        n_jobs = None
    
    # Always use random_seed from global config for rng_seed (syncs with command-line seed)
    rng_seed = config.get('random_seed')
    
    # Get output_register and set output_registers if not explicitly provided
    output_register = eval_cfg.get('output_register', 0)
    output_registers = eval_cfg.get('output_registers')
    if output_registers is None:
        # Derive from output_register
        from memory_system import MemoryType
        output_registers = [(MemoryType.SCALAR, output_register)]
    
    return FlappyBirdEvaluatorConfig(
        env_id=eval_cfg.get('env_id', 'FlappyBird-v0'),
        episodes=eval_cfg.get('episodes', 5),
        max_steps=eval_cfg.get('max_steps', 500),
        output_register=output_register,
        render_mode=render_mode,
        rng_seed=rng_seed,
        patch_strategy=eval_cfg.get('patch_strategy', 'quantized'),
        color_channel=eval_cfg.get('color_channel', 1),
        normalize=eval_cfg.get('normalize', True),
        quantization_factor=eval_cfg.get('quantization_factor', 0.5),
        feature_vector_size=eval_cfg.get('feature_vector_size', 64),
        frame_stack_size=eval_cfg.get('frame_stack_size', 1),
        # Trinary strategy parameters
        trinary_crop_bottom=eval_cfg.get('trinary_crop_bottom', 100),
        trinary_resize_factor=eval_cfg.get('trinary_resize_factor', 0.03),
        trinary_final_size=eval_cfg.get('trinary_final_size', 21),
        trinary_bird_h_min=eval_cfg.get('trinary_bird_h_min', 0),
        trinary_bird_h_max=eval_cfg.get('trinary_bird_h_max', 50),
        trinary_bird_s_min=eval_cfg.get('trinary_bird_s_min', 50),
        trinary_bird_s_max=eval_cfg.get('trinary_bird_s_max', 255),
        trinary_bird_v_min=eval_cfg.get('trinary_bird_v_min', 50),
        trinary_bird_v_max=eval_cfg.get('trinary_bird_v_max', 255),
        trinary_pipe_h_min=eval_cfg.get('trinary_pipe_h_min', 35),
        trinary_pipe_h_max=eval_cfg.get('trinary_pipe_h_max', 45),
        trinary_pipe_s_min=eval_cfg.get('trinary_pipe_s_min', 40),
        trinary_pipe_s_max=eval_cfg.get('trinary_pipe_s_max', 255),
        trinary_pipe_v_min=eval_cfg.get('trinary_pipe_v_min', 40),
        trinary_pipe_v_max=eval_cfg.get('trinary_pipe_v_max', 255),
        n_jobs=n_jobs,
        output_registers=output_registers,  # Set output_registers explicitly
    )


def create_population_config(config: Dict[str, Any]) -> PopulationConfig:
    """Create PopulationConfig from YAML config.
    
    Args:
        config: Dictionary containing configuration sections
        
    Returns:
        PopulationConfig object
        
    Raises:
        ValueError: If required fields are missing or invalid
    """
    pop_cfg = config.get('population', {})
    
    # Validate required fields
    if 'size' not in pop_cfg:
        raise ValueError("Missing required population configuration field: size")
    if 'program_length' not in pop_cfg:
        raise ValueError("Missing required population configuration field: program_length")
    
    # Convert program_length from list to tuple
    program_length = pop_cfg.get('program_length', [1, 10])
    if isinstance(program_length, list):
        if len(program_length) != 2:
            raise ValueError(f"program_length must be a 2-element list, got: {program_length}")
        program_length = tuple(program_length)
    elif not isinstance(program_length, tuple):
        raise ValueError(f"program_length must be a list or tuple, got: {type(program_length)}")
    
    return PopulationConfig(
        size=pop_cfg['size'],
        program_length=program_length,
        elitism=pop_cfg.get('elitism', 1),
        max_program_length=pop_cfg.get('max_program_length'),
    )


def create_evolution_config(
    config: Dict[str, Any],
    manager: Optional['ExperimentManager'] = None,
) -> EvolutionConfig:
    """Create EvolutionConfig from YAML config.
    
    Args:
        config: Dictionary containing configuration sections
        manager: Optional ExperimentManager to provide output paths
        
    Returns:
        EvolutionConfig object
    """
    evo_cfg = config.get('evolution', {})
    exp_cfg = config.get('experiment', {})
    
    # Get checkpoint and stats paths from manager if provided
    if manager is not None:
        # Use manager's paths
        checkpoint_dir = str(manager.checkpoints_dir)
        stats_log_path = str(manager.get_stats_csv_path())
        checkpoint_every = manager.checkpoint_every
    else:
        # No manager - disable checkpoints and stats logging
        checkpoint_dir = None
        stats_log_path = None
        checkpoint_every = exp_cfg.get('checkpoint_every')
    
    return EvolutionConfig(
        max_generations=evo_cfg.get('max_generations', 100),
        mutation_threshold=evo_cfg.get('mutation_threshold', 0.1),
        constant_mutation_rate=evo_cfg.get('constant_mutation_rate', 0.0),
        crossover_threshold=evo_cfg.get('crossover_threshold', 0.9),
        verbose=evo_cfg.get('verbose', True),
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=checkpoint_every,
        stats_log_path=stats_log_path,
    )


def get_operations_config(config: Dict[str, Any]) -> Dict[str, bool]:
    """Get operations configuration.
    
    Args:
        config: Dictionary containing configuration sections
        
    Returns:
        Dictionary with operation set boolean flags (checked in order, first true wins)
    """
    ops_cfg = config.get('operations', {})
    return {
        'use_feature_vector_ops': ops_cfg.get('use_feature_vector_ops', False),
        'use_minimal_scalar': ops_cfg.get('use_minimal_scalar', False),
        'use_minimal': ops_cfg.get('use_minimal', False),
        'use_automl_no_random': ops_cfg.get('use_automl_no_random', False),
        'use_automl': ops_cfg.get('use_automl', True),
        'use_cv': ops_cfg.get('use_cv', True),
    }


def get_experiment_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get experiment configuration.
    
    Args:
        config: Dictionary containing configuration sections
        
    Returns:
        Dictionary with experiment settings for ExperimentManager
    """
    exp_cfg = config.get('experiment', {})
    
    # Handle null values
    name = exp_cfg.get('name')
    if name == 'null':
        name = None
    
    keep_n = exp_cfg.get('keep_n_checkpoints')
    if keep_n == 'null':
        keep_n = None
    
    checkpoint_every = exp_cfg.get('checkpoint_every')
    if checkpoint_every == 'null':
        checkpoint_every = None
    
    return {
        'name': name,
        'base_dir': exp_cfg.get('base_dir', 'experiments'),
        'checkpoint_every': checkpoint_every,
        'keep_n_checkpoints': keep_n,
    }
