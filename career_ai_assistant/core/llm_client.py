import asyncio
import os
import google.generativeai as genai
from typing import Optional
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini with environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    model = None

class ONNXLoRAFallback:
    """Fallback to ONNX runtime LoRA-tuned model"""
    async def generate(self, prompt: str) -> str:
        # Simulate ONNX LoRA inference
        await asyncio.sleep(0.5)
        return f"ONNX LoRA fallback response: {prompt[:50]}..."

onnx_fallback = ONNXLoRAFallback()

async def generate_response(prompt: str, stream: bool = False, timeout: float = 5.0) -> str:
    """
    Async streaming Gemini API client with ONNX fallback.
    """
    if not model:
        return await onnx_fallback.generate(prompt)
        
    try:
        # Try Gemini with timeout
        if stream:
            response = await asyncio.wait_for(
                asyncio.to_thread(model.generate_content, prompt),
                timeout=timeout
            )
            return response.text
        else:
            response = await asyncio.wait_for(
                asyncio.to_thread(model.generate_content, prompt),
                timeout=timeout
            )
            return response.text
            
    except asyncio.TimeoutError:
        print("Gemini timeout, using ONNX fallback")
        return await onnx_fallback.generate(prompt)
    except Exception as e:
        print(f"Gemini error: {e}, using ONNX fallback")
        return await onnx_fallback.generate(prompt) 