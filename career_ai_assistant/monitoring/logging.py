import asyncio
import logging
import time
from typing import Dict, Optional
from datetime import datetime

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('career_ai.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    def __init__(self):
        self.request_times = []
        self.error_count = 0
        self.fallback_count = 0
        
    async def log_request(self, query: str, response_time: float, source: str, cached: bool = False):
        """Log request performance"""
        self.request_times.append(response_time)
        
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'query': query[:100],  # Truncate long queries
            'response_time': response_time,
            'source': source,
            'cached': cached
        }
        
        if response_time > 2.0:
            logger.warning(f"Slow request: {response_time:.2f}s - {query[:50]}")
        else:
            logger.info(f"Request processed: {response_time:.2f}s via {source}")
            
    async def log_error(self, error: Exception, context: str):
        """Log errors"""
        self.error_count += 1
        logger.error(f"Error in {context}: {str(error)}")
        
    async def log_fallback(self, original_source: str, fallback_source: str):
        """Log fallback usage"""
        self.fallback_count += 1
        logger.warning(f"Fallback from {original_source} to {fallback_source}")
        
    async def get_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.request_times:
            return {
                'avg_response_time': 0,
                'total_requests': 0,
                'error_count': self.error_count,
                'fallback_count': self.fallback_count
            }
            
        return {
            'avg_response_time': sum(self.request_times) / len(self.request_times),
            'total_requests': len(self.request_times),
            'error_count': self.error_count,
            'fallback_count': self.fallback_count,
            'slow_requests': len([t for t in self.request_times if t > 2.0])
        }

performance_monitor = PerformanceMonitor() 