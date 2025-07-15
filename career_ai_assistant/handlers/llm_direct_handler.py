import asyncio
import google.generativeai as genai
from typing import Dict
import os
import sys
from pathlib import Path
import time

# Add the parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from monitoring.logging import logger
from core.rag_engine import rag_engine

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

async def handle_query(query: str) -> Dict:
    """
    Ultra-fast direct LLM processing with RAG context.
    Target: Sub-1.5 second responses.
    """
    start_time = time.time()
    
    try:
        # Initialize Gemini model with correct name
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Get RAG context in parallel
        context_task = asyncio.create_task(
            rag_engine.retrieve_relevant_context(query)
        )
        
        # Simple, fast prompt for quick responses
        prompt = f"""
You are Venkateswara Rao Narra, a Full Stack Python Developer with 4+ years of experience building real-world web apps and ML-based features.

User asked: "{query}"

Respond as Venkateswara in first person. Be helpful, concise, and include relevant technical details or code examples when appropriate.

Keep your response under 200 words unless the user asks for detailed technical information.
"""
        
        # Generate response with strict timeout
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    model.generate_content,
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        top_p=0.9,
                        top_k=40,
                        max_output_tokens=400,  # Reduced for faster responses
                    )
                ),
                timeout=1.5  # 1.5 second timeout for direct LLM
            )
            
            response_text = response.text.strip()
            
            # Wait for context (should be fast)
            try:
                context = await asyncio.wait_for(context_task, timeout=0.2)
                # Enhance response with context if needed
                if len(response_text) < 100:
                    response_text = f"{context} {response_text}"
            except asyncio.TimeoutError:
                pass  # Continue without context enhancement
            
        except asyncio.TimeoutError:
            logger.warning(f"Direct LLM timeout for: {query[:30]}...")
            # Fallback to simple response
            response_text = f"""
Hi! I'm Venkateswara Rao Narra, a Full Stack Python Developer with 4+ years of experience in building real-world web apps and ML-based features.

I'd be happy to help you with "{query}"! I specialize in Python, React, TypeScript, FastAPI, Django, AWS, and ML applications.

What would you like to know about my skills, projects, or experience?
"""
        
        response_time = time.time() - start_time
        
        # Log performance
        if response_time < 1.5:
            logger.info(f"Fast direct LLM: {response_time:.2f}s")
        else:
            logger.warning(f"Slow direct LLM: {response_time:.2f}s")
        
        return {'response': response_text}
        
    except Exception as e:
        logger.error(f"Error in direct LLM handler: {e}")
        response_time = time.time() - start_time
        
        # Fast fallback response
        fallback_response = f"""
Hi! I'm Venkateswara Rao Narra, a Full Stack Python Developer with 4+ years of experience. 

I'd be happy to help you with "{query}"! I have expertise in:
- Full-stack development (Python, React, TypeScript)
- Cloud platforms (AWS, Azure)
- ML/AI applications and APIs
- Microservices and scalable systems

What would you like to know?
"""
        return {'response': fallback_response} 