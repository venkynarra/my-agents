import asyncio
import google.generativeai as genai
from typing import Dict
import os
import sys
from pathlib import Path

# Add the parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from monitoring.logging import logger
from agents.mcp_agent import get_mcp_response

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

async def handle_query(query: str) -> Dict:
    """
    Route queries through MCP agent for enhanced retrieval, optimized for speed.
    Follows the diagram flow: MCP Tools + Enhanced Knowledge + LLM.
    """
    try:
        # Get enhanced knowledge from MCP agent with timeout
        try:
            mcp_result = await asyncio.wait_for(
                get_mcp_response(query),
                timeout=3.0  # 3 second timeout for MCP
            )
            knowledge_context = mcp_result.get('context', '')
        except asyncio.TimeoutError:
            logger.warning(f"MCP timeout for query: {query}")
            knowledge_context = ""
        
        # Initialize Gemini model for enhanced response generation
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Simplified prompt for faster responses
        prompt = f"""
You are Venkateswara Rao Narra, a Full Stack Python Developer with 4+ years of experience.

**Context:** {knowledge_context}

**User Query:** {query}

Respond as Venkateswara in first person. Be helpful and concise. Include relevant technical details or code examples when appropriate.

Keep your response under 300 words unless the user asks for detailed technical information.
"""

        # Generate enhanced response with timeout
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    model.generate_content,
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        top_p=0.9,
                        top_k=40,
                        max_output_tokens=800,  # Reduced for faster responses
                    )
                ),
                timeout=5.0  # 5 second timeout
            )
            
            response_text = response.text.strip()
            
        except asyncio.TimeoutError:
            logger.warning(f"Gemini API timeout for MCP query: {query}")
            # Fallback to simple response
            response_text = f"""
Hi! I'm Venkateswara Rao Narra, a Full Stack Python Developer with 4+ years of experience.

I'd be happy to help you with "{query}"! I specialize in full-stack development, cloud architecture, and ML/AI applications.

What would you like to know about my skills, projects, or experience?
"""
        
        logger.info(f"Generated enhanced MCP response for query: {query[:50]}...")
        return {'response': response_text}
        
    except Exception as e:
        logger.error(f"Error in MCP router: {e}")
        # Fallback to direct LLM response
        try:
            from handlers.llm_direct_handler import handle_query as direct_handle
            return await direct_handle(query)
        except:
            # Ultimate fallback
            fallback_response = f"""
Hi! I'm Venkateswara Rao Narra, a Full Stack Python Developer with 4+ years of experience.

I'd be happy to help you with "{query}"! I have extensive experience in:
- Full-stack development with modern technologies
- Cloud architecture and scalable systems
- ML/AI applications and integrations
- DevOps and automation

What would you like to know?
"""
            return {'response': fallback_response} 