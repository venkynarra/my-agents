import yaml
import os
from pathlib import Path

def get_config():
    """Load configuration from YAML file"""
    config_path = Path(__file__).parent / "settings.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Replace environment variable placeholders
    _replace_env_vars(config)
    
    return config

def _replace_env_vars(config_dict):
    """Recursively replace environment variable placeholders in config"""
    for key, value in config_dict.items():
        if isinstance(value, dict):
            _replace_env_vars(value)
        elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]  # Remove ${ and }
            config_dict[key] = os.getenv(env_var, value) 