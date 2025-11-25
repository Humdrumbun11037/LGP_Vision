"""Configuration loader for YAML-based configuration."""

import yaml
from pathlib import Path
from typing import Optional, Dict, Any

from memory_system import MemoryConfig
from evaluator import FlappyBirdEvaluatorConfig
from population import PopulationConfig
from evolution_engine import EvolutionConfig


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
    
    # Use random_seed from global config if rng_seed not specified
    rng_seed = eval_cfg.get('rng_seed')
    if rng_seed is None:
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
        feature_vector_size=eval_cfg.get('feature_vector_size', 256),
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


def create_evolution_config(config: Dict[str, Any]) -> EvolutionConfig:
    """Create EvolutionConfig from YAML config.
    
    Args:
        config: Dictionary containing configuration sections
        
    Returns:
        EvolutionConfig object
        
    Raises:
        ValueError: If required fields are missing or invalid
    """
    evo_cfg = config.get('evolution', {})
    
    # Handle null checkpoint_path and stats_log_dir
    checkpoint_path = evo_cfg.get('checkpoint_path')
    if checkpoint_path == 'null' or checkpoint_path is None:
        checkpoint_path = None
    
    stats_log_dir = evo_cfg.get('stats_log_dir')
    if stats_log_dir == 'null' or stats_log_dir is None:
        stats_log_dir = None
    
    return EvolutionConfig(
        max_generations=evo_cfg.get('max_generations', 100),
        mutation_threshold=evo_cfg.get('mutation_threshold', 0.1),
        constant_mutation_rate=evo_cfg.get('constant_mutation_rate', 0.0),
        crossover_threshold=evo_cfg.get('crossover_threshold', 0.9),
        verbose=evo_cfg.get('verbose', True),
        checkpoint_path=checkpoint_path,
        stats_log_dir=stats_log_dir,
    )


def get_operations_config(config: Dict[str, Any]) -> Dict[str, bool]:
    """Get operations configuration.
    
    Args:
        config: Dictionary containing configuration sections
        
    Returns:
        Dictionary with 'use_automl' and 'use_cv' boolean flags
    """
    ops_cfg = config.get('operations', {})
    return {
        'use_automl': ops_cfg.get('use_automl', True),
        'use_cv': ops_cfg.get('use_cv', True),
    }


def get_output_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get output configuration.
    
    Args:
        config: Dictionary containing configuration sections
        
    Returns:
        Dictionary with output settings
    """
    return config.get('output', {
        'save_best_agent': True,
        'best_agent_dir': 'best_agents',
        'fitness_chart_path': 'fitness_chart.png',
    })

