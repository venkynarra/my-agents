"""
Enhanced Memory Caching Layer for AI Assistant
Provides intelligent caching for queries, embeddings, and enriched prompts with similarity matching.
Target: <50ms cache lookups, >80% cache hit rate for similar queries
"""
import json
import logging
import hashlib
import pickle
import asyncio
import threading
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cached item with metadata."""
    key: str
    value: Any
    embedding: Optional[np.ndarray] = None
    created_at: datetime = None
    accessed_at: datetime = None
    access_count: int = 0
    similarity_threshold: float = 0.85
    ttl_seconds: int = 3600  # 1 hour default
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if not self.created_at:
            return True
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds

@dataclass
class CacheStats:
    """Cache performance statistics."""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    similarity_matches: int = 0
    avg_lookup_time_ms: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        return (self.cache_hits / max(self.total_requests, 1)) * 100

class EnhancedMemoryCache:
    """
    Enhanced in-memory caching system with similarity matching and performance optimization.
    Provides fast caching without external Redis dependency.
    """
    
    def __init__(self, max_size: int = 10000, similarity_model: str = "all-MiniLM-L6-v2"):
        self.max_size = max_size
        self.similarity_model_name = similarity_model
        self._cache: Dict[str, CacheEntry] = {}
        self._similarity_index: Dict[str, Tuple[str, np.ndarray]] = {}
        self._lock = threading.RLock()
        self._stats = CacheStats()
        self._embedding_model = None
        self._initialized = False
        
        logger.info(f"📦 Enhanced Memory Cache initialized (max_size: {max_size})")
    
    async def initialize(self) -> bool:
        """Initialize embedding model."""
        try:
            logger.info("🚀 Initializing Enhanced Memory Cache...")
            
            # Initialize embedding model
            try:
                self._embedding_model = SentenceTransformer(self.similarity_model_name)
                logger.info(f"✅ Embedding model loaded: {self.similarity_model_name}")
            except Exception as e:
                logger.warning(f"Could not load embedding model: {e}")
                self._embedding_model = None
            
            self._initialized = True
            logger.info("✅ Enhanced Memory Cache initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Memory cache: {e}")
            return False
    
    async def get_query_response(self, query: str, 
                               similarity_threshold: float = 0.85) -> Optional[Dict[str, Any]]:
        """Get cached response for a query with similarity matching."""
        start_time = time.time()
        
        try:
            with self._lock:
                self._stats.total_requests += 1
                
                # Clean expired entries first
                self._cleanup_expired()
                
                # Direct cache hit
                cache_key = self._generate_query_key(query)
                if cache_key in self._cache:
                    entry = self._cache[cache_key]
                    if not entry.is_expired():
                        entry.accessed_at = datetime.now()
                        entry.access_count += 1
                        self._stats.cache_hits += 1
                        
                        lookup_time = (time.time() - start_time) * 1000
                        self._update_performance_stats(lookup_time)
                        
                        logger.debug(f"🎯 Direct cache hit for query: {query[:50]}...")
                        return entry.value
                
                # Similarity search if embedding model available
                if self._embedding_model:
                    similar_result = await self._find_similar_query(query, similarity_threshold)
                    if similar_result:
                        self._stats.similarity_matches += 1
                        self._stats.cache_hits += 1
                        
                        lookup_time = (time.time() - start_time) * 1000
                        self._update_performance_stats(lookup_time)
                        
                        logger.debug(f"🔍 Similarity match for query: {query[:50]}...")
                        return similar_result
                
                # Cache miss
                self._stats.cache_misses += 1
                lookup_time = (time.time() - start_time) * 1000
                self._update_performance_stats(lookup_time)
                
                logger.debug(f"❌ Cache miss for query: {query[:50]}...")
                return None
                
        except Exception as e:
            logger.error(f"Error getting cached query response: {e}")
            return None
    
    async def cache_query_response(self, query: str, response: Dict[str, Any], 
                                 ttl_seconds: int = 3600):
        """Cache a query response with automatic similarity indexing."""
        try:
            with self._lock:
                cache_key = self._generate_query_key(query)
                
                # Generate embedding for similarity search
                embedding = None
                if self._embedding_model:
                    try:
                        embedding = self._embedding_model.encode([query])[0]
                        await self._add_to_similarity_index(query, cache_key, embedding)
                    except Exception as e:
                        logger.warning(f"Could not generate embedding for query: {e}")
                
                # Create cache entry
                entry = CacheEntry(
                    key=cache_key,
                    value=response,
                    embedding=embedding,
                    created_at=datetime.now(),
                    accessed_at=datetime.now(),
                    access_count=1,
                    ttl_seconds=ttl_seconds
                )
                
                # Ensure cache doesn't exceed max size
                if len(self._cache) >= self.max_size:
                    self._evict_lru_entries()
                
                self._cache[cache_key] = entry
                logger.debug(f"💾 Cached query response: {query[:50]}...")
                
        except Exception as e:
            logger.error(f"Error caching query response: {e}")
    
    async def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding for text."""
        try:
            with self._lock:
                embedding_key = self._generate_embedding_key(text)
                
                if embedding_key in self._cache:
                    entry = self._cache[embedding_key]
                    if not entry.is_expired():
                        entry.accessed_at = datetime.now()
                        entry.access_count += 1
                        return entry.value
                
                # Generate new embedding if model available
                if self._embedding_model:
                    embedding = self._embedding_model.encode([text])[0]
                    
                    # Cache the embedding
                    entry = CacheEntry(
                        key=embedding_key,
                        value=embedding,
                        created_at=datetime.now(),
                        accessed_at=datetime.now(),
                        ttl_seconds=7200  # 2 hours for embeddings
                    )
                    
                    if len(self._cache) >= self.max_size:
                        self._evict_lru_entries()
                    
                    self._cache[embedding_key] = entry
                    return embedding
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None
    
    async def cache_enriched_prompt(self, original_query: str, enriched_prompt: str, 
                                  intent: str, ttl_seconds: int = 1800):
        """Cache an enriched prompt."""
        try:
            with self._lock:
                prompt_key = f"prompt:{intent}:{self._generate_query_key(original_query)}"
                
                entry = CacheEntry(
                    key=prompt_key,
                    value=enriched_prompt,
                    created_at=datetime.now(),
                    accessed_at=datetime.now(),
                    ttl_seconds=ttl_seconds
                )
                
                if len(self._cache) >= self.max_size:
                    self._evict_lru_entries()
                
                self._cache[prompt_key] = entry
                logger.debug(f"💾 Cached enriched prompt for intent: {intent}")
                
        except Exception as e:
            logger.error(f"Error caching enriched prompt: {e}")
    
    async def get_enriched_prompt(self, query: str, intent: str) -> Optional[str]:
        """Get cached enriched prompt."""
        try:
            with self._lock:
                prompt_key = f"prompt:{intent}:{self._generate_query_key(query)}"
                
                if prompt_key in self._cache:
                    entry = self._cache[prompt_key]
                    if not entry.is_expired():
                        entry.accessed_at = datetime.now()
                        entry.access_count += 1
                        return entry.value
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting enriched prompt: {e}")
            return None
    
    async def _find_similar_query(self, query: str, threshold: float) -> Optional[Dict[str, Any]]:
        """Find similar cached query using embedding similarity."""
        try:
            if not self._embedding_model or not self._similarity_index:
                return None
            
            query_embedding = self._embedding_model.encode([query])[0]
            
            best_similarity = 0.0
            best_match = None
            
            for indexed_query, (cache_key, stored_embedding) in self._similarity_index.items():
                if cache_key in self._cache and not self._cache[cache_key].is_expired():
                    similarity = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        stored_embedding.reshape(1, -1)
                    )[0][0]
                    
                    if similarity > best_similarity and similarity >= threshold:
                        best_similarity = similarity
                        best_match = self._cache[cache_key].value
            
            return best_match
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return None
    
    async def _add_to_similarity_index(self, query: str, cache_key: str, embedding: np.ndarray):
        """Add query embedding to similarity index."""
        try:
            # Limit similarity index size
            if len(self._similarity_index) >= 1000:
                # Remove oldest entries
                keys_to_remove = list(self._similarity_index.keys())[:100]
                for key in keys_to_remove:
                    del self._similarity_index[key]
            
            self._similarity_index[query] = (cache_key, embedding)
            
        except Exception as e:
            logger.error(f"Error adding to similarity index: {e}")
    
    def _generate_query_key(self, query: str) -> str:
        """Generate cache key for query."""
        return f"query:{hashlib.md5(query.encode()).hexdigest()}"
    
    def _generate_embedding_key(self, text: str) -> str:
        """Generate cache key for embedding."""
        return f"embedding:{hashlib.md5(text.encode()).hexdigest()}"
    
    def _cleanup_expired(self):
        """Remove expired entries from cache."""
        expired_keys = []
        for key, entry in self._cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
            # Also remove from similarity index
            for query, (cache_key, _) in list(self._similarity_index.items()):
                if cache_key == key:
                    del self._similarity_index[query]
                    break
    
    def _evict_lru_entries(self):
        """Evict least recently used entries."""
        # Sort by last access time and remove oldest 10%
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: x[1].accessed_at or datetime.min
        )
        
        evict_count = max(1, len(sorted_entries) // 10)
        for i in range(evict_count):
            key, _ = sorted_entries[i]
            del self._cache[key]
            
            # Also remove from similarity index
            for query, (cache_key, _) in list(self._similarity_index.items()):
                if cache_key == key:
                    del self._similarity_index[query]
                    break
    
    def _update_performance_stats(self, lookup_time_ms: float):
        """Update performance statistics."""
        # Moving average
        if self._stats.avg_lookup_time_ms == 0:
            self._stats.avg_lookup_time_ms = lookup_time_ms
        else:
            self._stats.avg_lookup_time_ms = (
                self._stats.avg_lookup_time_ms * 0.9 + lookup_time_ms * 0.1
            )
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            cache_sizes = {
                "total_entries": len(self._cache),
                "similarity_index_size": len(self._similarity_index),
                "max_size": self.max_size
            }
            
            stats = {
                "performance": asdict(self._stats),
                "cache_sizes": cache_sizes,
                "utilization": {
                    "cache_utilization_percent": (len(self._cache) / self.max_size) * 100,
                    "embedding_model_available": self._embedding_model is not None
                },
                "health": {
                    "status": "healthy" if self._initialized else "not_initialized",
                    "total_entries": len(self._cache)
                }
            }
            
            return stats
    
    async def clear_cache(self, cache_type: Optional[str] = None):
        """Clear cache entries."""
        with self._lock:
            if cache_type is None:
                self._cache.clear()
                self._similarity_index.clear()
                logger.info("🗑️ All cache cleared")
            else:
                # Clear specific cache type
                keys_to_remove = [k for k in self._cache.keys() if k.startswith(cache_type)]
                for key in keys_to_remove:
                    del self._cache[key]
                logger.info(f"🗑️ {cache_type} cache cleared ({len(keys_to_remove)} entries)")
    
    async def close(self):
        """Close the cache."""
        with self._lock:
            self._cache.clear()
            self._similarity_index.clear()
        logger.info("✅ Enhanced Memory Cache closed")

# Alias for compatibility
EnhancedRedisCache = EnhancedMemoryCache

async def create_enhanced_cache(redis_url: str = None) -> EnhancedMemoryCache:
    """Create enhanced memory cache instance."""
    logger.info("🚀 Creating Enhanced Memory Cache...")
    
    cache = EnhancedMemoryCache()
    success = await cache.initialize()
    
    if success:
        logger.info("✅ Enhanced Memory Cache created successfully")
        return cache
    else:
        logger.error("❌ Enhanced Memory Cache initialization failed")
        return None 