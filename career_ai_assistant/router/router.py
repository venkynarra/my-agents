import asyncio
import time
from typing import Dict
import sys
from pathlib import Path

# Add the parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from router.preprocessor import preprocess_query
from core.grpc_server import career_service
from agents.mcp_agent import get_mcp_response  # This now uses the enhanced MCP agent with tools
from monitoring.logging import logger

async def route_query(query: str) -> dict:
    """
    High-performance query routing using gRPC service and MCP agent.
    Target: Sub-2 second responses with proper flow.
    """
    start_time = time.time()
    
    try:
        # Step 1: Query Classification (from diagram)
        preproc = await preprocess_query(query)
        
        # Step 2: Route based on classification
        # Only truly simple/greeting queries go to direct LLM; all else use enhanced MCP agent
        if preproc['route_direct'] and preproc['intent'] in ['simple', 'greeting', 'casual']:
            result = await career_service.process_query(query, use_cache=True)
            source = 'llm_direct_handler'
        else:
            mcp_result = await get_mcp_response(query)
            result = {
                'response': mcp_result['response'],
                'cached': False
            }
            source = 'mcp_router'
        
        response_time = time.time() - start_time
        
        # Log performance
        if response_time < 2.0:
            logger.info(f"Fast routing: {response_time:.2f}s - {source} - Intent: {preproc['intent']}")
        else:
            logger.warning(f"Slow routing: {response_time:.2f}s - {source} - Intent: {preproc['intent']}")
        
        return {
            'source': source, 
            'response': result['response'], 
            'cached': result.get('cached', False),
            'response_time': response_time
        }
        
    except Exception as e:
        logger.error(f"Routing error: {e}")
        response_time = time.time() - start_time
        
        # Fallback to MCP agent first, then direct service
        try:
            mcp_result = await get_mcp_response(query)
            return {
                'source': 'mcp_fallback', 
                'response': mcp_result['response'], 
                'cached': False,
                'response_time': response_time
            }
        except:
            try:
                result = await career_service.process_query(query, use_cache=True)
                return {
                    'source': 'grpc_fallback', 
                    'response': result['response'], 
                    'cached': False,
                    'response_time': response_time
                }
            except:
                # Ultimate fallback
                fallback_response = f"""
Hi! I'm Venkateswara Rao Narra, a Full Stack Python Developer with 4+ years of experience.

I'd be happy to help you with "{query}"! I have expertise in:
- Full-stack development (Python, React, TypeScript)
- Cloud platforms (AWS, Azure)
- ML/AI applications and APIs
- Microservices and scalable systems

What would you like to know?
"""
                return {
                    'source': 'ultimate_fallback', 
                    'response': fallback_response, 
                    'cached': False,
                    'response_time': response_time
                } 