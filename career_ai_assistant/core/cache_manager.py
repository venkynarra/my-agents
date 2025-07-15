import redis
import asyncio
import json
from typing import Dict, Optional, List
import math
import sys
from pathlib import Path

# Add the parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import get_config

config = get_config()
REDIS_URL = config['redis']['url']

class CacheManager:
    def __init__(self):
        self.redis_client = None
        self.connection_failed = False
        
    async def connect(self):
        """Connect to Redis with better error handling"""
        if self.connection_failed:
            return  # Don't retry if we've already failed
            
        try:
            self.redis_client = redis.from_url(
                REDIS_URL,
                socket_connect_timeout=config['redis']['socket_connect_timeout'],
                socket_timeout=config['redis']['socket_timeout'],
                retry_on_timeout=config['redis']['retry_on_timeout']
            )
            # Test connection with timeout
            await asyncio.wait_for(
                asyncio.to_thread(self.redis_client.ping),
                timeout=config['redis']['timeout']
            )
            print("Redis connection successful")
        except Exception as e:
            print(f"Redis connection failed: {e}")
            self.redis_client = None
            self.connection_failed = True
        
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            try:
                await asyncio.to_thread(self.redis_client.close)
            except Exception as e:
                print(f"Redis close error: {e}")
            
    async def get_cache(self, key: str) -> Optional[str]:
        """Get cached response"""
        if not self.redis_client and not self.connection_failed:
            await self.connect()
        if self.redis_client:
            try:
                result = await asyncio.to_thread(self.redis_client.get, key)
                return result.decode() if result else None
            except Exception as e:
                print(f"Redis get error: {e}")
                return None
        return None
        
    async def set_cache(self, key: str, value: str, expire: int = 3600):
        """Set cached response"""
        if not self.redis_client and not self.connection_failed:
            await self.connect()
        if self.redis_client:
            try:
                await asyncio.to_thread(self.redis_client.set, key, value, ex=expire)
            except Exception as e:
                print(f"Redis set error: {e}")
        
    async def store_embedding(self, query: str, embedding: List[float], response: str):
        """Store query embedding and response"""
        if not self.redis_client and not self.connection_failed:
            await self.connect()
        if self.redis_client:
            try:
                embedding_key = f"emb:{hash(query)}"
                data = {
                    'embedding': embedding,
                    'query': query,
                    'response': response
                }
                await asyncio.to_thread(self.redis_client.set, embedding_key, json.dumps(data), ex=3600)
            except Exception as e:
                print(f"Redis embedding store error: {e}")
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
            
        return dot_product / (magnitude1 * magnitude2)
        
    async def similarity_search(self, query_embedding: List[float], threshold: float = 0.8) -> Optional[str]:
        """Find similar cached responses using cosine similarity"""
        if not self.redis_client and not self.connection_failed:
            await self.connect()
        if not self.redis_client:
            return None
            
        try:
            # Get all embedding keys
            keys = await asyncio.to_thread(self.redis_client.keys, "emb:*")
            best_match = None
            best_score = 0
            
            for key in keys:
                cached_data = await asyncio.to_thread(self.redis_client.get, key)
                if cached_data:
                    data = json.loads(cached_data)
                    cached_embedding = data['embedding']
                    
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(query_embedding, cached_embedding)
                    
                    if similarity > threshold and similarity > best_score:
                        best_score = similarity
                        best_match = data['response']
                        
            return best_match
        except Exception as e:
            print(f"Redis similarity search error: {e}")
            return None

cache_manager = CacheManager() 