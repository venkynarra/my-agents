"""
Enhanced Multi-Agent Router with Performance Optimization
Coordinates multiple AI components for sub-2s response times with intelligent routing.
"""
import asyncio
import logging
import time
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import enhanced components with absolute imports
try:
    from preprocessing_layer import QueryAnalysis, RoutingDecision, create_preprocessor
except ImportError:
    from foundations.preprocessing_layer import QueryAnalysis, RoutingDecision, create_preprocessor

try:
    from enhanced_cache import EnhancedRedisCache, create_enhanced_cache
except ImportError:
    from foundations.enhanced_cache import EnhancedRedisCache, create_enhanced_cache

try:
    from pinecone_engine import EnhancedPineconeEngine, create_pinecone_engine
except ImportError:
    from foundations.pinecone_engine import EnhancedPineconeEngine, create_pinecone_engine

try:
    from llm_precorrector import LLMPreCorrector, EnhancedQuery, create_llm_precorrector
except ImportError:
    from foundations.llm_precorrector import LLMPreCorrector, EnhancedQuery, create_llm_precorrector

try:
    from prompt_optimizer import PromptOptimizer, OptimizedPrompt, create_prompt_optimizer
except ImportError:
    from foundations.prompt_optimizer import PromptOptimizer, OptimizedPrompt, create_prompt_optimizer

try:
    from rag_engine import create_llm_client
except ImportError:
    from foundations.rag_engine import create_llm_client

logger = logging.getLogger(__name__)

@dataclass
class ResponseMetrics:
    """Detailed response generation metrics."""
    total_time_ms: float
    preprocessing_time_ms: float
    cache_lookup_time_ms: float
    knowledge_retrieval_time_ms: float
    llm_generation_time_ms: float
    cache_hit: bool
    components_used: List[str]
    optimization_applied: bool

class EnhancedMultiAgentRouter:
    """
    Enhanced multi-agent router with performance optimization and robust error handling.
    Coordinates preprocessing, caching, enhancement, and generation for <2s responses.
    """
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379",
                 pinecone_api_key: Optional[str] = None,
                 gemini_api_key: Optional[str] = None):
        
        # Configuration - load from environment if not provided
        self.redis_url = redis_url
        self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        
        # Component instances
        self.preprocessor = None
        self.cache = None
        self.vector_engine = None
        self.precorrector = None
        self.prompt_optimizer = None
        self.llm = None
        
        # Performance tracking
        self._performance_stats = {
            "total_requests": 0,
            "avg_response_time_ms": 0.0,
            "sub_2s_rate": 0.0,
            "cache_hit_rate": 0.0,
            "error_rate": 0.0
        }
        self._response_times = []
        self._errors = 0
        
        # Fallback responses for common queries
        self._fallback_responses = {
            "contact": """You can reach me through:
📧 Email: vnarrag@gmu.edu
📱 Phone: +1 703-453-2157
💼 LinkedIn: https://www.linkedin.com/in/venkateswara-narra-91170b34a
🔗 GitHub: https://github.com/venkynarra

I'm always open to discussing new opportunities and technical collaborations!""",
            
            "default": """I'm Venkatesh Narra, a Software Development Engineer with 4+ years of experience in AI/ML and full-stack development. 
I specialize in building scalable systems and have worked on projects including AI testing agents, clinical APIs, and multi-modal chat platforms. 
Feel free to ask me about my technical skills, experience, or projects!"""
        }

    async def initialize(self) -> bool:
        """Initialize all components with optimized startup."""
        try:
            start_time = time.time()
            logger.info("🚀 Initializing Enhanced Multi-Agent Router...")
            
            # Initialize components in parallel where possible
            init_tasks = []
            
            # Critical path components (sequential)
            await self._init_preprocessor()
            await self._init_cache()
            
            # Parallel initialization for other components
            vector_task = asyncio.create_task(self._init_vector_engine())
            precorrector_task = asyncio.create_task(self._init_precorrector())
            prompt_task = asyncio.create_task(self._init_prompt_optimizer())
            llm_task = asyncio.create_task(self._init_llm())
            
            # Wait for parallel initialization
            await asyncio.gather(vector_task, precorrector_task, prompt_task, llm_task, return_exceptions=True)
            
            init_time = (time.time() - start_time) * 1000
            logger.info(f"✅ Enhanced Router initialized in {init_time:.1f}ms")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Router initialization failed: {e}")
            return False

    async def _init_preprocessor(self):
        self.preprocessor = await create_preprocessor()
        logger.info("✅ preprocessor initialized successfully")

    async def _init_cache(self):
        self.cache = await create_enhanced_cache(self.redis_url)
        logger.info("✅ cache initialized successfully")

    async def _init_vector_engine(self):
        try:
            self.vector_engine = await create_pinecone_engine(self.pinecone_api_key)
            logger.info("✅ vector_engine initialized successfully")
        except Exception as e:
            logger.warning(f"Vector engine initialization failed: {e}")

    async def _init_precorrector(self):
        try:
            self.precorrector = await create_llm_precorrector(self.gemini_api_key)
            logger.info("✅ precorrector initialized successfully")
        except Exception as e:
            logger.warning(f"Precorrector initialization failed: {e}")

    async def _init_prompt_optimizer(self):
        try:
            self.prompt_optimizer = create_prompt_optimizer()
            logger.info("✅ prompt_optimizer initialized successfully")
        except Exception as e:
            logger.warning(f"Prompt optimizer initialization failed: {e}")

    async def _init_llm(self):
        try:
            self.llm = await create_llm_client(self.gemini_api_key)
            logger.info("✅ llm initialized successfully")
        except Exception as e:
            logger.warning(f"LLM initialization failed: {e}")

    async def route_query(self, query: str, use_cache: bool = True) -> str:
        """
        Route query through enhanced pipeline with robust error handling.
        Target: <2000ms response time with high reliability.
        """
        start_time = time.time()
        
        # Initialize metrics
        metrics = ResponseMetrics(
            total_time_ms=0.0,
            preprocessing_time_ms=0.0,
            cache_lookup_time_ms=0.0,
            knowledge_retrieval_time_ms=0.0,
            llm_generation_time_ms=0.0,
            cache_hit=False,
            components_used=[],
            optimization_applied=False
        )
        
        try:
            # Step 1: Fast preprocessing
            preprocessing_start = time.time()
            query_analysis = await self._analyze_query_fast(query)
            metrics.preprocessing_time_ms = (time.time() - preprocessing_start) * 1000
            metrics.components_used.append("preprocessor")
            
            # Fast path for simple contact queries
            if self._is_contact_query(query):
                return self._fallback_responses["contact"]
            
            # Step 2: Cache lookup with timeout
            cache_result = None
            if use_cache and self.cache:
                try:
                    cache_result = await asyncio.wait_for(
                        self._check_cache_async(query, query_analysis),
                        timeout=0.1  # 100ms timeout for cache
                    )
                    if cache_result:
                        metrics.cache_hit = True
                        metrics.total_time_ms = (time.time() - start_time) * 1000
                        self._update_performance_stats(metrics)
                        return cache_result.get("response", "Cached response not available")
                except asyncio.TimeoutError:
                    logger.warning("Cache lookup timeout")
                except Exception as e:
                    logger.warning(f"Cache lookup error: {e}")
            
            # Step 3: Enhanced processing with timeout
            try:
                enhanced_query = await asyncio.wait_for(
                    self._enhance_query_async(query, query_analysis),
                    timeout=1.5  # 1.5s timeout for enhancement
                )
            except asyncio.TimeoutError:
                logger.warning("Query enhancement timeout, using fallback")
                enhanced_query = self._create_fallback_enhanced_query(query)
            except Exception as e:
                logger.warning(f"Query enhancement error: {e}")
                enhanced_query = self._create_fallback_enhanced_query(query)
            
            # Step 4: Parallel knowledge retrieval and prompt optimization with timeouts
            try:
                knowledge_task = asyncio.wait_for(
                    self._retrieve_knowledge_async(enhanced_query, query_analysis),
                    timeout=0.3  # 300ms timeout
                )
                prompt_task = asyncio.wait_for(
                    self._optimize_prompt_async(enhanced_query, query_analysis),
                    timeout=0.2  # 200ms timeout
                )
                
                knowledge_context, optimized_prompt = await asyncio.gather(
                    knowledge_task, 
                    prompt_task,
                    return_exceptions=True
                )
                
                # Handle exceptions from parallel tasks
                if isinstance(knowledge_context, Exception):
                    logger.warning(f"Knowledge retrieval failed: {knowledge_context}")
                    knowledge_context = {}
                
                if isinstance(optimized_prompt, Exception):
                    logger.warning(f"Prompt optimization failed: {optimized_prompt}")
                    optimized_prompt = self._create_fallback_prompt(enhanced_query)
                
                metrics.optimization_applied = True
                metrics.components_used.extend(["precorrector", "prompt_optimizer"])
                
            except Exception as e:
                logger.error(f"Error in parallel processing: {e}")
                knowledge_context = {}
                optimized_prompt = self._create_fallback_prompt(enhanced_query)
            
            # Step 5: LLM generation with timeout
            llm_start = time.time()
            try:
                response = await asyncio.wait_for(
                    self._generate_response_async(optimized_prompt, knowledge_context),
                    timeout=2.0  # 2s timeout for LLM
                )
            except asyncio.TimeoutError:
                logger.warning("LLM generation timeout")
                response = await self._generate_fallback_response(query)
            except Exception as e:
                logger.error(f"LLM generation error: {e}")
                response = await self._generate_fallback_response(query)
            
            metrics.llm_generation_time_ms = (time.time() - llm_start) * 1000
            metrics.components_used.append("llm")
            
            # Step 6: Cache the response (fire and forget)
            if self.cache and use_cache and len(response) > 50:  # Only cache substantial responses
                asyncio.create_task(self._cache_response_safe(query, response, enhanced_query))
            
            # Update metrics and stats
            metrics.total_time_ms = (time.time() - start_time) * 1000
            self._update_performance_stats(metrics)
            
            if metrics.total_time_ms > 2000:
                logger.warning(f"⚠️ Response in {metrics.total_time_ms:.1f}ms (>2s)")
            else:
                logger.info(f"✅ Response in {metrics.total_time_ms:.1f}ms (sub-2s)")
            
            return response
            
        except Exception as e:
            self._errors += 1
            logger.error(f"Error in enhanced routing: {e}")
            # Ultimate fallback
            return await self._generate_fallback_response(query)

    def _is_contact_query(self, query: str) -> bool:
        """Quick check for contact-related queries."""
        contact_keywords = ["contact", "reach", "email", "phone", "linkedin", "github", "call", "message"]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in contact_keywords)

    def _create_fallback_enhanced_query(self, query: str):
        """Create a fallback enhanced query."""
        from .llm_precorrector import EnhancedQuery, QueryIntent, QueryComplexity
        return EnhancedQuery(
            original_query=query,
            corrected_query=query,
            intent=QueryIntent.CONVERSATIONAL,
            complexity=QueryComplexity.MODERATE,
            added_keywords=[],
            confidence=0.5,
            suggested_followups=[],
            processing_notes="Fallback enhancement",
            estimated_response_time=1.0
        )

    def _create_fallback_prompt(self, enhanced_query):
        """Create a fallback optimized prompt."""
        from .prompt_optimizer import OptimizedPrompt, PromptTemplate, ResponseStyle
        return OptimizedPrompt(
            prompt=f"You are Venkatesh Narra. Question: {enhanced_query.corrected_query}\n\nResponse:",
            template_used=PromptTemplate.CONVERSATIONAL,
            style_used=ResponseStyle.PROFESSIONAL,
            estimated_tokens=100,
            context_included=[],
            optimization_notes="Fallback prompt"
        )

    async def _analyze_query_fast(self, query: str) -> QueryAnalysis:
        """Fast query analysis using preprocessing layer."""
        if self.preprocessor:
            try:
                return await self.preprocessor.analyze_query(query)
            except Exception as e:
                logger.warning(f"Preprocessing failed: {e}")
        
        # Fallback analysis
        from .preprocessing_layer import QueryAnalysis, RoutingDecision
        return QueryAnalysis(
            intent="conversational",
            routing=RoutingDecision.ENHANCED_PIPELINE,
            confidence=0.5,
            keywords=[],
            complexity="medium",
            estimated_response_time=1.5,
            cache_key=None,
            suggested_template="conversational"
        )

    async def _check_cache_async(self, query: str, analysis: QueryAnalysis) -> Optional[Dict[str, Any]]:
        """Asynchronous cache lookup."""
        if not self.cache:
            return None
            
        try:
            cache_start = time.time()
            result = await self.cache.get_query_response(query, similarity_threshold=0.85)
            cache_time = (time.time() - cache_start) * 1000
            
            if result:
                logger.debug(f"⚡ Cache hit in {cache_time:.1f}ms")
                return result
            return None
        except Exception as e:
            logger.warning(f"Cache lookup failed: {e}")
            return None

    async def _enhance_query_async(self, query: str, analysis: QueryAnalysis):
        """Asynchronous query enhancement."""
        if self.precorrector:
            try:
                return await self.precorrector.enhance_query(query)
            except Exception as e:
                logger.warning(f"Query enhancement failed: {e}")
        
        return self._create_fallback_enhanced_query(query)

    async def _retrieve_knowledge_async(self, enhanced_query, analysis: QueryAnalysis) -> Dict[str, Any]:
        """Asynchronous knowledge retrieval from multiple sources."""
        knowledge_context = {}
        
        try:
            # Vector search if available (with timeout)
            if self.vector_engine:
                try:
                    vector_results = await asyncio.wait_for(
                        self.vector_engine.semantic_search(
                            enhanced_query.corrected_query,
                            top_k=2,  # Reduced for speed
                            min_score=0.7
                        ),
                        timeout=0.2  # 200ms timeout
                    )
                    knowledge_context["vector_results"] = [
                        {"content": result.content[:200], "score": result.score}  # Truncated for speed
                        for result in vector_results[:2]  # Top 2 only
                    ]
                except asyncio.TimeoutError:
                    logger.warning("Vector search timeout")
                except Exception as e:
                    logger.warning(f"Vector search failed: {e}")
            
            # Static knowledge as fallback/supplement
            knowledge_context["static_context"] = self._get_static_knowledge_context(enhanced_query.intent.value)
            
        except Exception as e:
            logger.warning(f"Knowledge retrieval failed: {e}")
            knowledge_context["static_context"] = "I'm Venkatesh Narra, a Software Development Engineer."
        
        return knowledge_context

    async def _optimize_prompt_async(self, enhanced_query, analysis: QueryAnalysis):
        """Asynchronous prompt optimization."""
        if self.prompt_optimizer:
            try:
                context_categories = self._determine_context_categories(enhanced_query)
                return self.prompt_optimizer.optimize_prompt(
                    query=enhanced_query.corrected_query,
                    intent=enhanced_query.intent.value,
                    context_categories=context_categories,
                    max_tokens=800,  # Reduced for speed
                    include_metrics=False  # Disable metrics for speed
                )
            except Exception as e:
                logger.warning(f"Prompt optimization failed: {e}")
        
        return self._create_fallback_prompt(enhanced_query)

    async def _generate_response_async(self, optimized_prompt, knowledge_context: Dict[str, Any]) -> str:
        """Asynchronous LLM response generation."""
        if not self.llm:
            return self._fallback_responses["default"]
        
        try:
            # Incorporate knowledge context into prompt (simplified)
            enhanced_prompt = self._incorporate_knowledge_context(optimized_prompt.prompt, knowledge_context)
            
            # Generate response
            response = await self.llm.acomplete(enhanced_prompt)
            return str(response).strip()
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return await self._generate_fallback_response(optimized_prompt.prompt)

    async def _cache_response_safe(self, original_query: str, response: str, enhanced_query):
        """Safe response caching with error handling."""
        try:
            if self.cache:
                await self.cache.cache_query_response(
                    query=original_query,
                    response={"response": response, "enhanced_query": enhanced_query.corrected_query},
                    ttl_seconds=3600
                )
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")

    def _get_static_knowledge_context(self, intent: str) -> str:
        """Get static knowledge context based on intent."""
        context_map = {
            "technical_skills": "Python, Java, JavaScript, AI/ML, FastAPI, React, AWS, Docker",
            "experience": "4+ years, Software Development Engineer at Veritis Group Inc",
            "projects": "AI testing agent, Clinical APIs, Multi-modal chat platform",
            "education": "MS Computer Science - George Mason University",
            "contact": "vnarrag@gmu.edu, +1 703-453-2157, LinkedIn, GitHub",
            "default": "Software Development Engineer with AI/ML expertise"
        }
        return context_map.get(intent, context_map["default"])

    def _determine_context_categories(self, enhanced_query) -> List[str]:
        """Determine context categories for prompt optimization."""
        intent = enhanced_query.intent.value.lower()
        
        category_map = {
            "technical_skills": ["skills", "technical"],
            "experience": ["experience", "professional"],
            "projects": ["projects", "technical"],
            "education": ["education", "academic"],
            "contact": ["contact", "personal"],
            "default": ["general"]
        }
        
        return category_map.get(intent, category_map["default"])

    def _incorporate_knowledge_context(self, prompt: str, knowledge_context: Dict[str, Any]) -> str:
        """Incorporate knowledge context into prompt efficiently."""
        if not knowledge_context:
            return prompt
        
        context_parts = []
        
        # Add static context
        if "static_context" in knowledge_context:
            context_parts.append(f"Key Info: {knowledge_context['static_context']}")
        
        # Add top vector result if available
        if "vector_results" in knowledge_context and knowledge_context["vector_results"]:
            top_result = knowledge_context["vector_results"][0]
            context_parts.append(f"Context: {top_result['content'][:150]}...")
        
        if context_parts:
            context_str = " | ".join(context_parts)
            return f"{context_str}\n\n{prompt}"
        
        return prompt

    async def _generate_fallback_response(self, query: str) -> str:
        """Generate fallback response for errors."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["contact", "reach", "email", "phone"]):
            return self._fallback_responses["contact"]
        elif any(word in query_lower for word in ["skills", "technical", "programming"]):
            return "I have 4+ years of experience with Python, Java, JavaScript, AI/ML frameworks like TensorFlow and PyTorch, cloud platforms like AWS, and full-stack development with React and FastAPI."
        elif any(word in query_lower for word in ["experience", "work", "job"]):
            return "I'm currently a Software Development Engineer at Veritis Group Inc with 4+ years of experience. I've built AI testing agents, clinical APIs handling 25,000+ daily inferences, and multi-modal chat platforms."
        else:
            return self._fallback_responses["default"]

    def _update_performance_stats(self, metrics: ResponseMetrics):
        """Update performance statistics."""
        self._performance_stats["total_requests"] += 1
        self._response_times.append(metrics.total_time_ms)
        
        # Keep only last 100 response times
        if len(self._response_times) > 100:
            self._response_times = self._response_times[-100:]
        
        # Update averages
        self._performance_stats["avg_response_time_ms"] = sum(self._response_times) / len(self._response_times)
        self._performance_stats["sub_2s_rate"] = (
            sum(1 for t in self._response_times if t < 2000) / len(self._response_times)
        ) * 100
        
        if metrics.cache_hit:
            cache_hits = sum(1 for _ in range(min(100, self._performance_stats["total_requests"])))
            self._performance_stats["cache_hit_rate"] = (cache_hits / min(100, self._performance_stats["total_requests"])) * 100

    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            "performance": self._performance_stats.copy(),
            "recent_response_times": self._response_times[-10:],
            "component_status": {
                "preprocessor": self.preprocessor is not None,
                "cache": self.cache is not None,
                "vector_engine": self.vector_engine is not None,
                "precorrector": self.precorrector is not None,
                "prompt_optimizer": self.prompt_optimizer is not None,
                "llm": self.llm is not None
            }
        }

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check."""
        components = {
            "preprocessor": self.preprocessor is not None,
            "cache": self.cache is not None,
            "vector_engine": self.vector_engine is not None,
            "precorrector": self.precorrector is not None,
            "prompt_optimizer": self.prompt_optimizer is not None,
            "llm": self.llm is not None
        }
        
        available_count = sum(components.values())
        
        return {
            "status": "healthy" if available_count >= 4 else "degraded",
            "components": components,
            "available_count": available_count,
            "total_components": len(components),
            "performance": self._performance_stats,
            "error_rate": (self._errors / max(1, self._performance_stats["total_requests"])) * 100
        }

    async def close(self):
        """Close all components and cleanup."""
        logger.info("🛑 Shutting down Enhanced Multi-Agent Router...")
        
        cleanup_tasks = []
        if self.cache:
            cleanup_tasks.append(self.cache.close())
        if self.vector_engine:
            cleanup_tasks.append(self.vector_engine.close())
        
        if cleanup_tasks:
            try:
                results = await asyncio.gather(*cleanup_tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                        logger.warning(f"⚠️ Cleanup task failed: {result}")
            except asyncio.CancelledError:
                logger.info("🔄 Cleanup cancelled - continuing graceful shutdown")
            except Exception as e:
                logger.warning(f"⚠️ Error during cleanup: {e}")
        
        logger.info("✅ Enhanced Router shutdown complete")

async def create_enhanced_router(redis_url: str = "redis://localhost:6379",
                               pinecone_api_key: Optional[str] = None,
                               gemini_api_key: Optional[str] = None) -> EnhancedMultiAgentRouter:
    """Create and initialize the enhanced multi-agent router."""
    logger.info("🚀 Creating Enhanced Multi-Agent Router...")
    
    router = EnhancedMultiAgentRouter(
        redis_url=redis_url,
        pinecone_api_key=pinecone_api_key,
        gemini_api_key=gemini_api_key
    )
    
    success = await router.initialize()
    if success:
        logger.info("✅ Enhanced Multi-Agent Router ready")
        return router
    else:
        logger.error("❌ Enhanced Multi-Agent Router initialization failed")
        return router  # Return anyway for partial functionality 