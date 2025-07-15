import asyncio
import grpc
from concurrent.futures import ThreadPoolExecutor
import time
from typing import Dict, Any
import sys
from pathlib import Path
import google.generativeai as genai
import os

# Add the parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from monitoring.logging import logger
from core.rag_engine import rag_engine

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

class CareerAssistantService:
    """High-performance career assistant service with gRPC"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.response_cache = {}  # In-memory cache for ultra-fast responses
        self.request_times = {}
        
    async def process_query(self, query: str, use_cache: bool = True) -> Dict[str, Any]:
        """Process query with performance optimization and guardrails"""
        start_time = time.time()
        
        try:
            # Check in-memory cache first for ultra-fast responses
            if use_cache and query.lower().strip() in self.response_cache:
                cached_response = self.response_cache[query.lower().strip()]
                response_time = time.time() - start_time
                logger.info(f"Ultra-fast cache hit: {response_time:.3f}s")
                return {
                    'response': cached_response,
                    'source': 'ultra_cache',
                    'response_time': response_time,
                    'cached': True
                }
            
            # Query classification and routing
            query_type = self._classify_query(query)
            
            if query_type == 'simple':
                # Direct LLM for simple queries (fastest path)
                result = await self._fast_direct_llm(query)
            else:
                # Enhanced processing for complex queries
                result = await self._fast_enhanced_processing(query)
            
            response_time = time.time() - start_time
            
            # Guardrails: Ensure response quality
            result = self._apply_guardrails(result, query)
            
            # Cache successful responses
            if response_time < 2.0 and result.get('response'):
                self.response_cache[query.lower().strip()] = result['response']
            
            # Log performance
            if response_time > 2.0:
                logger.warning(f"Slow response: {response_time:.2f}s - {query[:50]}...")
            else:
                logger.info(f"Fast response: {response_time:.2f}s - {query[:50]}...")
            
            return {
                'response': result['response'],
                'source': result.get('source', 'unknown'),
                'response_time': response_time,
                'cached': False
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            response_time = time.time() - start_time
            return {
                'response': self._get_dynamic_fallback_response(query),
                'source': 'fallback',
                'response_time': response_time,
                'cached': False
            }
    
    def _classify_query(self, query: str) -> str:
        """Fast query classification"""
        query_lower = query.lower()
        
        # Simple queries that can be handled directly
        simple_keywords = ['hi', 'hello', 'hey']
        if any(keyword in query_lower for keyword in simple_keywords):
            return 'simple'
        
        # Complex queries that need enhanced knowledge
        complex_keywords = ['experience', 'project', 'certification', 'skill', 'code', 'technical', 'about', 'who are you', 'intro']
        if any(keyword in query_lower for keyword in complex_keywords):
            return 'complex'
        
        return 'complex'  # Default to complex for better responses
    
    async def _fast_direct_llm(self, query: str) -> Dict[str, Any]:
        """Ultra-fast direct LLM processing with enhanced system prompt and QKV values"""
        try:
            # Initialize Gemini model
            model = genai.GenerativeModel('gemini-1.5-pro')
            
            # Enhanced system prompt for greetings/simple queries
            prompt = f"""
You are Venkateswara Rao Narra, a Full Stack Python Developer with 4+ years of experience.
ROLE: Senior software engineer, expert in Python, React, ML, and cloud.
PERSONALITY: Confident, helpful, detailed, always in first person.
RESPONSE STYLE: Detailed, structured, with real examples, code, and quantifiable achievements.

User said: "{query}"

Give a warm, personal greeting and briefly mention your expertise. Be specific about what you do. Always answer as Venkatesh in first person. If the user asks about your background, skills, or experience, provide a short but concrete summary with at least one real project or achievement.
"""
            
            # Use asyncio.wait_for for strict timeout
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    model.generate_content,
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.92,  # More creative
                        top_p=0.97,
                        top_k=60,
                        max_output_tokens=600,  # More tokens for richer responses
                    )
                ),
                timeout=1.5  # 1.5 second timeout
            )
            
            response_text = response.text.strip()
            return {'response': response_text, 'source': 'direct_llm'}
            
        except asyncio.TimeoutError:
            logger.warning(f"Direct LLM timeout for: {query[:30]}...")
            return {'response': self._get_dynamic_fallback_response(query), 'source': 'fallback'}
        except Exception as e:
            logger.error(f"Direct LLM error: {e}")
            return {'response': self._get_dynamic_fallback_response(query), 'source': 'fallback'}

    async def _fast_enhanced_processing(self, query: str) -> Dict[str, Any]:
        """Fast enhanced processing with RAG context and elaborate system prompt/QKV values"""
        try:
            # Get RAG context
            context = await asyncio.wait_for(
                rag_engine.retrieve_relevant_context(query),
                timeout=0.5
            )
            
            # Initialize Gemini model
            model = genai.GenerativeModel('gemini-1.5-pro')
            
            # Enhanced system prompt for complex queries
            prompt = f"""
You are Venkateswara Rao Narra, a Full Stack Python Developer with 4+ years of experience.
ROLE: Senior software engineer, expert in Python, React, ML, and cloud.
PERSONALITY: Confident, helpful, detailed, always in first person.
RESPONSE STYLE: Detailed, structured, with real examples, code, and quantifiable achievements.

**Context:** {context}

**User Query:** {query}

Respond as Venkatesh in first person. Use the context to provide specific, detailed answers. Include concrete examples from your actual experience and projects. Be comprehensive, structured, and helpful. If relevant, include code snippets, technical details, and quantifiable results. Always mention real projects, technologies, and achievements.

Target length: 400-600 words for complex queries.
"""
            
            # Generate response with timeout
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    model.generate_content,
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.92,  # More creative
                        top_p=0.97,
                        top_k=60,
                        max_output_tokens=1200,  # More tokens for richer responses
                    )
                ),
                timeout=2.0  # 2 second timeout for enhanced processing
            )
            
            response_text = response.text.strip()
            return {'response': response_text, 'source': 'enhanced_processing'}
            
        except asyncio.TimeoutError:
            logger.warning(f"Enhanced processing timeout for: {query[:30]}...")
            # Fallback to direct LLM
            return await self._fast_direct_llm(query)
        except Exception as e:
            logger.error(f"Enhanced processing error: {e}")
            return await self._fast_direct_llm(query)
    
    def _apply_guardrails(self, result: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Apply response quality guardrails"""
        response = result.get('response', '')
        
        # Ensure response is not empty
        if not response or len(response.strip()) < 10:
            result['response'] = self._get_dynamic_fallback_response(query)
            return result
        
        # Ensure response is not too long (performance)
        if len(response) > 1000:
            result['response'] = response[:1000] + "..."
        
        return result
    
    def _get_dynamic_fallback_response(self, query: str) -> str:
        """Dynamic fallback response based on query type"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['hi', 'hello', 'hey']):
            return "Hi there! I'm Venkateswara Rao Narra, a Full Stack Python Developer with 4+ years of experience. How can I help you today?"
        
        elif any(word in query_lower for word in ['skill', 'technology', 'tech']):
            return "My technical skills include Python, Java, JavaScript, TypeScript, React, Angular, FastAPI, Django, Flask, AWS, Azure, Docker, Kubernetes, PostgreSQL, MongoDB, Redis, Git, CI/CD, Scikit-learn, Pandas, and TensorFlow. I have 4+ years of experience building real-world web apps and ML-based features."
        
        elif any(word in query_lower for word in ['experience', 'work', 'job', 'career']):
            return "I have 4+ years of experience as a software engineer. Currently working as Software Development Engineer at Veritis Group Inc (2023-Present), previously Full Stack Developer at TCS (2021-2022), and Junior Software Engineer at Virtusa (2020-2021). I've worked on clinical decision support tools, loan platforms, and retail reporting systems."
        
        elif any(word in query_lower for word in ['project', 'build', 'create', 'develop']):
            return "My key projects include: Clinical Decision Support Tool (FastAPI/React), Loan Platform (Django/React), Real-time ML Prediction APIs, Patient Risk Overview Dashboard, Music Streaming Service, and Stock Market Prediction using LSTM. I specialize in full-stack development with Python, React, and cloud technologies."
        
        elif any(word in query_lower for word in ['certification', 'cert', 'certified']):
            return "My certifications include: Advanced Learning Algorithms (Stanford), Artificial Intelligence I (IBM), and Deep Learning Specialization (Coursera). I also have a Master of Computer Science from George Mason University with a 3.47/4.00 GPA."
        
        else:
            return f"Hi! I'm Venkateswara Rao Narra, a Full Stack Python Developer with 4+ years of experience building real-world web apps and ML-based features. I'd be happy to help you with '{query}'! What would you like to know about my skills, projects, or experience?"

# Global service instance
career_service = CareerAssistantService() 