"""
LoRA and ONNX Local Model Support for AI Assistant
Provides local model fallback with LoRA fine-tuning capabilities and ONNX quantization.
Target: Sub-1s inference for simple queries with optional enhancement layer
"""
import asyncio
import logging
import os
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import time

# Core ML libraries
try:
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel, LoraConfig, get_peft_model
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# ONNX Runtime
try:
    import onnxruntime as ort
    import numpy as np
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for local models."""
    model_name: str
    model_type: str  # "transformers", "onnx", "lora"
    model_path: str
    tokenizer_path: Optional[str] = None
    max_length: int = 512
    temperature: float = 0.7
    device: str = "cpu"
    quantized: bool = False

@dataclass
class InferenceResult:
    """Result from local model inference."""
    text: str
    confidence: float
    inference_time_ms: float
    model_used: str
    tokens_generated: int

class LoRAEnhancementEngine:
    """
    Local model engine with LoRA fine-tuning and ONNX quantization support.
    Provides fast local inference as fallback or enhancement layer.
    """
    
    def __init__(self, 
                 models_dir: str = "./models",
                 device: str = "auto",
                 enable_quantization: bool = True):
        
        self.models_dir = Path(models_dir)
        self.device = self._detect_device(device)
        self.enable_quantization = enable_quantization
        
        # Model registry
        self.loaded_models: Dict[str, Any] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.tokenizers: Dict[str, Any] = {}
        
        # Performance tracking
        self.inference_stats = {
            "total_inferences": 0,
            "avg_inference_time": 0.0,
            "model_usage": {},
            "cache_hits": 0
        }
        
        # Response cache for fast repeated queries
        self.response_cache: Dict[str, InferenceResult] = {}
        self.cache_max_size = 100
        
        # Initialize default model configurations
        self._setup_default_models()
        
    def _detect_device(self, device: str) -> str:
        """Detect optimal device for inference."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"  # Apple Silicon
            else:
                return "cpu"
        return device
    
    def _setup_default_models(self):
        """Setup default model configurations."""
        self.model_configs = {
            "gemma-2b-quantized": ModelConfig(
                model_name="gemma-2b-quantized",
                model_type="onnx",
                model_path="google/gemma-2b",
                max_length=256,
                temperature=0.3,
                device=self.device,
                quantized=True
            ),
            "phi-3-mini": ModelConfig(
                model_name="phi-3-mini",
                model_type="transformers",
                model_path="microsoft/Phi-3-mini-4k-instruct",
                max_length=512,
                temperature=0.5,
                device=self.device,
                quantized=False
            ),
            "custom-lora": ModelConfig(
                model_name="custom-lora",
                model_type="lora",
                model_path="",  # To be set when LoRA adapter is available
                max_length=512,
                temperature=0.4,
                device=self.device,
                quantized=False
            )
        }
    
    async def initialize(self) -> bool:
        """Initialize the LoRA enhancement engine."""
        try:
            logger.info("🚀 Initializing LoRA Enhancement Engine...")
            
            if not TRANSFORMERS_AVAILABLE:
                logger.warning("⚠️ Transformers not available. LoRA features disabled.")
                return False
            
            # Create models directory
            self.models_dir.mkdir(exist_ok=True)
            
            # Try to load a lightweight model for testing
            success = await self._load_default_model()
            
            if success:
                logger.info("✅ LoRA Enhancement Engine initialized successfully")
                return True
            else:
                logger.warning("⚠️ LoRA Engine initialized with limited functionality")
                return True  # Still return True for graceful degradation
                
        except Exception as e:
            logger.error(f"❌ LoRA Engine initialization failed: {e}")
            return False
    
    async def _load_default_model(self) -> bool:
        """Load a default lightweight model for basic functionality."""
        try:
            # Try to load a small, fast model
            model_name = "microsoft/DialoGPT-small"  # Lightweight conversational model
            
            logger.info(f"📥 Loading default model: {model_name}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with optimization
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map="auto" if self.device != "cpu" else None,
                low_cpu_mem_usage=True
            )
            
            # Move to device
            if self.device == "cpu":
                model = model.to(self.device)
            
            # Store loaded components
            self.loaded_models["default"] = model
            self.tokenizers["default"] = tokenizer
            
            logger.info(f"✅ Default model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.warning(f"Could not load default model: {e}")
            return False
    
    async def load_lora_adapter(self, 
                              base_model: str,
                              adapter_path: str,
                              adapter_name: str = "custom") -> bool:
        """Load a LoRA adapter for fine-tuned inference."""
        try:
            if not TRANSFORMERS_AVAILABLE:
                logger.error("Transformers not available for LoRA loading")
                return False
            
            logger.info(f"📥 Loading LoRA adapter: {adapter_name}")
            
            # Load base model if not already loaded
            if base_model not in self.loaded_models:
                tokenizer = AutoTokenizer.from_pretrained(base_model)
                model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                    device_map="auto" if self.device != "cpu" else None
                )
                
                self.tokenizers[base_model] = tokenizer
                self.loaded_models[base_model] = model
            
            # Load LoRA adapter
            lora_model = PeftModel.from_pretrained(
                self.loaded_models[base_model],
                adapter_path
            )
            
            # Store LoRA model
            self.loaded_models[f"lora_{adapter_name}"] = lora_model
            self.tokenizers[f"lora_{adapter_name}"] = self.tokenizers[base_model]
            
            logger.info(f"✅ LoRA adapter '{adapter_name}' loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load LoRA adapter: {e}")
            return False
    
    async def create_onnx_model(self, 
                              model_name: str,
                              output_path: str,
                              quantize: bool = True) -> bool:
        """Convert a loaded model to ONNX format with optional quantization."""
        try:
            if not ONNX_AVAILABLE:
                logger.error("ONNX Runtime not available")
                return False
            
            if model_name not in self.loaded_models:
                logger.error(f"Model {model_name} not loaded")
                return False
            
            logger.info(f"🔄 Converting {model_name} to ONNX...")
            
            model = self.loaded_models[model_name]
            tokenizer = self.tokenizers[model_name]
            
            # Create dummy input for export
            dummy_input = tokenizer("Hello world", return_tensors="pt")
            
            # Export to ONNX
            torch.onnx.export(
                model,
                tuple(dummy_input.values()),
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input_ids', 'attention_mask'],
                output_names=['output'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence'},
                    'attention_mask': {0: 'batch_size', 1: 'sequence'},
                    'output': {0: 'batch_size', 1: 'sequence'}
                }
            )
            
            # Quantize if requested
            if quantize:
                await self._quantize_onnx_model(output_path)
            
            logger.info(f"✅ ONNX model saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"ONNX conversion failed: {e}")
            return False
    
    async def _quantize_onnx_model(self, model_path: str):
        """Quantize ONNX model for better performance."""
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            quantized_path = model_path.replace('.onnx', '_quantized.onnx')
            
            quantize_dynamic(
                model_path,
                quantized_path,
                weight_type=QuantType.QUInt8
            )
            
            logger.info(f"✅ Model quantized: {quantized_path}")
            
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
    
    async def enhance_query(self, 
                          query: str,
                          model_name: str = "default",
                          max_new_tokens: int = 100,
                          use_cache: bool = True) -> InferenceResult:
        """Enhance query using local model inference."""
        start_time = time.time()
        
        # Check cache first
        if use_cache and query in self.response_cache:
            cached_result = self.response_cache[query]
            self.inference_stats["cache_hits"] += 1
            logger.info(f"⚡ Cache hit for query: {query[:50]}...")
            return cached_result
        
        try:
            # Select model
            if model_name not in self.loaded_models:
                model_name = "default"
            
            if model_name not in self.loaded_models:
                return InferenceResult(
                    text="Local model not available",
                    confidence=0.0,
                    inference_time_ms=0.0,
                    model_used="none",
                    tokens_generated=0
                )
            
            model = self.loaded_models[model_name]
            tokenizer = self.tokenizers[model_name]
            
            # Prepare input
            input_text = self._prepare_input(query)
            inputs = tokenizer.encode(input_text, return_tensors="pt")
            
            if self.device != "cpu":
                inputs = inputs.to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs)
                )
            
            # Decode response
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_text = self._extract_response(generated_text, input_text)
            
            # Calculate metrics
            inference_time = (time.time() - start_time) * 1000
            tokens_generated = len(outputs[0]) - len(inputs[0])
            confidence = self._calculate_confidence(response_text, tokens_generated)
            
            # Create result
            result = InferenceResult(
                text=response_text,
                confidence=confidence,
                inference_time_ms=inference_time,
                model_used=model_name,
                tokens_generated=tokens_generated
            )
            
            # Cache result
            if use_cache:
                self._cache_result(query, result)
            
            # Update stats
            self._update_inference_stats(inference_time, model_name)
            
            logger.info(f"🤖 Local inference completed in {inference_time:.1f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Local inference failed: {e}")
            return InferenceResult(
                text=f"Local inference error: {str(e)}",
                confidence=0.0,
                inference_time_ms=(time.time() - start_time) * 1000,
                model_used=model_name,
                tokens_generated=0
            )
    
    def _prepare_input(self, query: str) -> str:
        """Prepare input text for the model."""
        # Simple prompt template for conversational response
        return f"Human: {query}\nAssistant:"
    
    def _extract_response(self, generated_text: str, input_text: str) -> str:
        """Extract the response from generated text."""
        # Remove the input prompt from generated text
        response = generated_text.replace(input_text, "").strip()
        
        # Clean up the response
        response = response.split("\nHuman:")[0].strip()  # Stop at next human input
        response = response.split("\n\n")[0].strip()  # Take first paragraph
        
        return response if response else "I'd be happy to help with that."
    
    def _calculate_confidence(self, text: str, tokens_generated: int) -> float:
        """Calculate confidence score for the generated response."""
        # Simple heuristics for confidence
        base_confidence = 0.7
        
        # Adjust based on response length
        if len(text) < 10:
            base_confidence -= 0.3
        elif len(text) > 50:
            base_confidence += 0.1
        
        # Adjust based on tokens generated
        if tokens_generated < 5:
            base_confidence -= 0.2
        elif tokens_generated > 20:
            base_confidence += 0.1
        
        return max(0.0, min(1.0, base_confidence))
    
    def _cache_result(self, query: str, result: InferenceResult):
        """Cache inference result."""
        if len(self.response_cache) >= self.cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self.response_cache))
            del self.response_cache[oldest_key]
        
        self.response_cache[query] = result
    
    def _update_inference_stats(self, inference_time: float, model_name: str):
        """Update inference statistics."""
        self.inference_stats["total_inferences"] += 1
        
        # Update average inference time
        current_avg = self.inference_stats["avg_inference_time"]
        total = self.inference_stats["total_inferences"]
        self.inference_stats["avg_inference_time"] = (
            (current_avg * (total - 1)) + inference_time
        ) / total
        
        # Update model usage
        self.inference_stats["model_usage"][model_name] = (
            self.inference_stats["model_usage"].get(model_name, 0) + 1
        )
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            "loaded_models": list(self.loaded_models.keys()),
            "model_configs": {name: {
                "model_name": config.model_name,
                "model_type": config.model_type,
                "device": config.device,
                "quantized": config.quantized
            } for name, config in self.model_configs.items()},
            "device": self.device,
            "inference_stats": self.inference_stats,
            "cache_size": len(self.response_cache)
        }
    
    async def benchmark_model(self, 
                            model_name: str = "default",
                            test_queries: Optional[List[str]] = None) -> Dict[str, Any]:
        """Benchmark a specific model's performance."""
        if test_queries is None:
            test_queries = [
                "What are your technical skills?",
                "Tell me about your experience.",
                "Why should we hire you?",
                "What projects have you worked on?"
            ]
        
        benchmark_results = {
            "model_name": model_name,
            "total_queries": len(test_queries),
            "results": [],
            "avg_inference_time": 0.0,
            "avg_confidence": 0.0
        }
        
        total_time = 0.0
        total_confidence = 0.0
        
        for query in test_queries:
            result = await self.enhance_query(query, model_name, use_cache=False)
            
            benchmark_results["results"].append({
                "query": query,
                "inference_time_ms": result.inference_time_ms,
                "confidence": result.confidence,
                "tokens_generated": result.tokens_generated
            })
            
            total_time += result.inference_time_ms
            total_confidence += result.confidence
        
        benchmark_results["avg_inference_time"] = total_time / len(test_queries)
        benchmark_results["avg_confidence"] = total_confidence / len(test_queries)
        
        logger.info(f"📊 Benchmark completed for {model_name}: {benchmark_results['avg_inference_time']:.1f}ms avg")
        return benchmark_results
    
    async def clear_cache(self):
        """Clear the response cache."""
        self.response_cache.clear()
        logger.info("🗑️ LoRA engine cache cleared")
    
    async def close(self):
        """Clean shutdown of the LoRA engine."""
        # Clear models from memory
        for model_name in list(self.loaded_models.keys()):
            del self.loaded_models[model_name]
        
        for tokenizer_name in list(self.tokenizers.keys()):
            del self.tokenizers[tokenizer_name]
        
        # Clear cache
        self.response_cache.clear()
        
        # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("✅ LoRA Enhancement Engine shutdown complete")


# Factory function
async def create_lora_engine(models_dir: str = "./models",
                           device: str = "auto",
                           enable_quantization: bool = True) -> Optional[LoRAEnhancementEngine]:
    """Create and initialize the LoRA enhancement engine."""
    logger.info("🚀 Creating LoRA Enhancement Engine...")
    
    engine = LoRAEnhancementEngine(
        models_dir=models_dir,
        device=device,
        enable_quantization=enable_quantization
    )
    
    success = await engine.initialize()
    if success:
        logger.info("✅ LoRA Enhancement Engine ready")
        return engine
    else:
        logger.warning("⚠️ LoRA Engine created with limited functionality")
        return engine  # Return even if limited for graceful degradation


# Utility functions
def get_available_models() -> Dict[str, str]:
    """Get list of available pre-trained models for local inference."""
    return {
        "microsoft/DialoGPT-small": "Lightweight conversational model",
        "microsoft/Phi-3-mini-4k-instruct": "Small instruction-following model",
        "google/gemma-2b": "Gemma 2B parameter model",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "Very small chat model",
        "distilbert-base-uncased": "Lightweight BERT variant"
    }

def estimate_model_requirements(model_name: str) -> Dict[str, Any]:
    """Estimate memory and compute requirements for a model."""
    requirements = {
        "microsoft/DialoGPT-small": {"memory_gb": 1.5, "inference_speed": "fast"},
        "microsoft/Phi-3-mini-4k-instruct": {"memory_gb": 2.5, "inference_speed": "medium"},
        "google/gemma-2b": {"memory_gb": 4.0, "inference_speed": "medium"},
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {"memory_gb": 2.0, "inference_speed": "fast"}
    }
    
    return requirements.get(model_name, {"memory_gb": "unknown", "inference_speed": "unknown"}) 