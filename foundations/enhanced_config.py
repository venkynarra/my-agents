"""
Enhanced Configuration for AI Assistant
Centralized configuration for all enhanced components with environment variable support.
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

@dataclass
class RedisConfig:
    """Redis cache configuration."""
    url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379"))
    password: Optional[str] = field(default_factory=lambda: os.getenv("REDIS_PASSWORD"))
    db: int = field(default_factory=lambda: int(os.getenv("REDIS_DB", "0")))
    max_connections: int = 20
    retry_on_timeout: bool = True

@dataclass
class PineconeConfig:
    """Pinecone vector database configuration."""
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("PINECONE_API_KEY"))
    environment: str = field(default_factory=lambda: os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws"))
    index_name: str = field(default_factory=lambda: os.getenv("PINECONE_INDEX_NAME", "career-assistant-knowledge"))
    dimension: int = 384
    metric: str = "cosine"

@dataclass
class GeminiConfig:
    """Gemini LLM configuration."""
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.3
    max_tokens: int = 1000
    timeout_seconds: float = 30.0

@dataclass
class LoRAConfig:
    """LoRA and local model configuration."""
    models_dir: str = field(default_factory=lambda: os.getenv("MODELS_DIR", "./models"))
    device: str = field(default_factory=lambda: os.getenv("MODEL_DEVICE", "auto"))
    enable_quantization: bool = field(default_factory=lambda: os.getenv("ENABLE_QUANTIZATION", "true").lower() == "true")
    default_model: str = "microsoft/DialoGPT-small"
    max_memory_gb: float = 8.0

@dataclass
class PreprocessingConfig:
    """Preprocessing layer configuration."""
    fuzzy_threshold: float = 70.0
    max_keywords: int = 5
    cache_size: int = 1000
    enable_metrics: bool = True

@dataclass
class PerformanceConfig:
    """Performance and optimization configuration."""
    target_response_time_ms: float = 2000.0
    cache_ttl_seconds: int = 3600
    max_concurrent_requests: int = 100
    thread_pool_workers: int = 4
    enable_parallel_processing: bool = True

@dataclass
class SecurityConfig:
    """Security and privacy configuration."""
    enable_logging: bool = field(default_factory=lambda: os.getenv("ENABLE_LOGGING", "true").lower() == "true")
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    mask_sensitive_data: bool = True
    rate_limit_requests_per_minute: int = 60

@dataclass
class EnhancedConfig:
    """Complete enhanced configuration for the AI assistant."""
    redis: RedisConfig = field(default_factory=RedisConfig)
    pinecone: PineconeConfig = field(default_factory=PineconeConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Server configuration
    grpc_port: int = field(default_factory=lambda: int(os.getenv("GRPC_PORT", "50051")))
    gradio_port: int = field(default_factory=lambda: int(os.getenv("GRADIO_PORT", "7860")))
    
    # Feature flags
    enable_cache: bool = field(default_factory=lambda: os.getenv("ENABLE_CACHE", "true").lower() == "true")
    enable_vector_search: bool = field(default_factory=lambda: os.getenv("ENABLE_VECTOR_SEARCH", "true").lower() == "true")
    enable_lora: bool = field(default_factory=lambda: os.getenv("ENABLE_LORA", "false").lower() == "true")
    enable_preprocessing: bool = field(default_factory=lambda: os.getenv("ENABLE_PREPROCESSING", "true").lower() == "true")
    
    def validate(self) -> Dict[str, Any]:
        """Validate configuration and return status."""
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        # Check required API keys
        if not self.gemini.api_key:
            validation_results["errors"].append("GEMINI_API_KEY is required")
            validation_results["valid"] = False
        
        # Check optional but recommended configurations
        if self.enable_vector_search and not self.pinecone.api_key:
            validation_results["warnings"].append("PINECONE_API_KEY not set - vector search will be disabled")
        
        if not self.redis.url:
            validation_results["warnings"].append("Redis URL not configured - caching will be limited")
        
        # Performance recommendations
        if self.performance.target_response_time_ms > 3000:
            validation_results["recommendations"].append("Consider reducing target response time for better user experience")
        
        if self.lora.max_memory_gb < 4.0 and self.enable_lora:
            validation_results["recommendations"].append("Consider increasing memory allocation for LoRA models")
        
        return validation_results
    
    def get_feature_summary(self) -> Dict[str, bool]:
        """Get summary of enabled features."""
        return {
            "enhanced_cache": self.enable_cache,
            "vector_search": self.enable_vector_search and bool(self.pinecone.api_key),
            "lora_support": self.enable_lora,
            "preprocessing": self.enable_preprocessing,
            "parallel_processing": self.performance.enable_parallel_processing,
            "quantization": self.lora.enable_quantization,
            "logging": self.security.enable_logging
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "redis": {
                "url": self.redis.url,
                "db": self.redis.db,
                "max_connections": self.redis.max_connections
            },
            "pinecone": {
                "environment": self.pinecone.environment,
                "index_name": self.pinecone.index_name,
                "dimension": self.pinecone.dimension
            },
            "gemini": {
                "model_name": self.gemini.model_name,
                "temperature": self.gemini.temperature,
                "max_tokens": self.gemini.max_tokens
            },
            "lora": {
                "models_dir": self.lora.models_dir,
                "device": self.lora.device,
                "enable_quantization": self.lora.enable_quantization
            },
            "performance": {
                "target_response_time_ms": self.performance.target_response_time_ms,
                "cache_ttl_seconds": self.performance.cache_ttl_seconds,
                "max_concurrent_requests": self.performance.max_concurrent_requests
            },
            "features": self.get_feature_summary()
        }


# Global configuration instance
config = EnhancedConfig()

# Legacy compatibility exports
GEMINI_API_KEY = config.gemini.api_key
REDIS_URL = config.redis.url
PINECONE_API_KEY = config.pinecone.api_key

# Environment detection
def get_environment() -> str:
    """Detect the current environment."""
    env = os.getenv("ENVIRONMENT", "development").lower()
    if env in ["prod", "production"]:
        return "production"
    elif env in ["staging", "stage"]:
        return "staging"
    elif env in ["test", "testing"]:
        return "testing"
    else:
        return "development"

# Configuration profiles for different environments
def get_config_for_environment(env: str = None) -> EnhancedConfig:
    """Get configuration optimized for specific environment."""
    if env is None:
        env = get_environment()
    
    base_config = EnhancedConfig()
    
    if env == "production":
        # Production optimizations
        base_config.performance.target_response_time_ms = 1500.0
        base_config.performance.max_concurrent_requests = 200
        base_config.performance.thread_pool_workers = 8
        base_config.security.log_level = "WARNING"
        base_config.redis.max_connections = 50
        
    elif env == "staging":
        # Staging configuration
        base_config.performance.target_response_time_ms = 2000.0
        base_config.performance.max_concurrent_requests = 50
        base_config.security.log_level = "INFO"
        
    elif env == "testing":
        # Testing configuration
        base_config.enable_cache = False
        base_config.enable_vector_search = False
        base_config.enable_lora = False
        base_config.security.log_level = "DEBUG"
        base_config.performance.target_response_time_ms = 5000.0
        
    else:  # development
        # Development configuration
        base_config.security.log_level = "DEBUG"
        base_config.performance.target_response_time_ms = 3000.0
        base_config.security.enable_logging = True
    
    return base_config

# Utility functions
def print_config_summary():
    """Print a summary of the current configuration."""
    print("🔧 Enhanced AI Assistant Configuration")
    print("=" * 50)
    
    validation = config.validate()
    if validation["valid"]:
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration has errors:")
        for error in validation["errors"]:
            print(f"  - {error}")
    
    if validation["warnings"]:
        print("\n⚠️ Warnings:")
        for warning in validation["warnings"]:
            print(f"  - {warning}")
    
    if validation["recommendations"]:
        print("\n💡 Recommendations:")
        for rec in validation["recommendations"]:
            print(f"  - {rec}")
    
    print(f"\n🚀 Environment: {get_environment()}")
    print(f"📊 Features enabled:")
    
    features = config.get_feature_summary()
    for feature, enabled in features.items():
        status = "✅" if enabled else "❌"
        print(f"  {status} {feature}")
    
    print(f"\n⚡ Performance target: <{config.performance.target_response_time_ms}ms")
    print(f"🔌 gRPC port: {config.grpc_port}")
    print(f"🌐 Gradio port: {config.gradio_port}")

def load_config_from_file(config_path: str) -> EnhancedConfig:
    """Load configuration from a JSON file."""
    import json
    
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Override environment variables with file values
        for section, values in config_data.items():
            if hasattr(config, section):
                section_config = getattr(config, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
        
        print(f"✅ Configuration loaded from {config_path}")
        return config
        
    except Exception as e:
        print(f"❌ Failed to load configuration from {config_path}: {e}")
        return config

def save_config_to_file(config_path: str):
    """Save current configuration to a JSON file."""
    import json
    
    try:
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        print(f"✅ Configuration saved to {config_path}")
        
    except Exception as e:
        print(f"❌ Failed to save configuration to {config_path}: {e}")

# Example configuration file generation
def create_example_config():
    """Create an example configuration file."""
    example_config = {
        "redis": {
            "url": "redis://localhost:6379",
            "password": None,
            "db": 0
        },
        "pinecone": {
            "api_key": "your-pinecone-api-key",
            "environment": "us-east-1-aws",
            "index_name": "career-assistant-knowledge"
        },
        "gemini": {
            "api_key": "your-gemini-api-key",
            "model_name": "gemini-1.5-flash",
            "temperature": 0.3
        },
        "performance": {
            "target_response_time_ms": 2000.0,
            "max_concurrent_requests": 100
        },
        "features": {
            "enable_cache": True,
            "enable_vector_search": True,
            "enable_lora": False,
            "enable_preprocessing": True
        }
    }
    
    return example_config

if __name__ == "__main__":
    print_config_summary() 