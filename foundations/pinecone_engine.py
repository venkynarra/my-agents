"""
Enhanced Memory Vector Database Engine for AI Assistant
Simple in-memory vector database for development without external dependencies.
Target: <200ms query time, support for 10k+ vectors
"""
import asyncio
import logging
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import json
import hashlib
import threading
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logger first
logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning("SentenceTransformers not available. Install with: pip install sentence-transformers")

from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class VectorSearchResult:
    """Result from vector search with metadata."""
    id: str
    score: float
    content: str
    metadata: Dict[str, Any]
    category: Optional[str] = None

@dataclass
class IndexStats:
    """Vector index statistics."""
    total_vectors: int
    index_size_bytes: int
    dimension: int
    avg_query_time_ms: float = 0.0
    total_queries: int = 0

class EnhancedMemoryVectorEngine:
    """
    Enhanced in-memory vector database with similarity search and category filtering.
    Provides vector storage and retrieval without external dependencies.
    """
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 dimension: int = 384,
                 max_vectors: int = 50000):
        self.embedding_model_name = embedding_model
        self.dimension = dimension
        self.max_vectors = max_vectors
        self._embedding_model = None
        self._vectors: Dict[str, np.ndarray] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._categories: Dict[str, List[str]] = {}
        self._lock = threading.RLock()
        self._stats = IndexStats(total_vectors=0, index_size_bytes=0, dimension=dimension)
        self._initialized = False
        
        logger.info(f"📦 Enhanced Memory Vector Engine initialized (max_vectors: {max_vectors})")
    
    async def initialize(self) -> bool:
        """Initialize the vector engine."""
        try:
            logger.info("🚀 Initializing Enhanced Memory Vector Engine...")
            
            # Initialize embedding model
            if EMBEDDINGS_AVAILABLE:
                try:
                    self._embedding_model = SentenceTransformer(self.embedding_model_name)
                    logger.info(f"✅ Embedding model loaded: {self.embedding_model_name}")
                except Exception as e:
                    logger.warning(f"Could not load embedding model: {e}")
                    self._embedding_model = None
            
            self._initialized = True
            logger.info("✅ Enhanced Memory Vector Engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Memory Vector Engine: {e}")
            return False
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to the vector index."""
        if not self._embedding_model:
            logger.warning("No embedding model available for document indexing")
            return False
        
        try:
            with self._lock:
                added_count = 0
                
                for doc in documents:
                    if len(self._vectors) >= self.max_vectors:
                        logger.warning(f"Maximum vector limit ({self.max_vectors}) reached")
                        break
                    
                    content = doc.get("content", "")
                    category = doc.get("category", "general")
                    metadata = doc.get("metadata", {})
                    
                    # Generate embedding
                    try:
                        embedding = self._embedding_model.encode([content])[0]
                        doc_id = self._generate_document_id(content, category)
                        
                        # Store vector and metadata
                        self._vectors[doc_id] = embedding
                        self._metadata[doc_id] = {
                            "content": content,
                            "category": category,
                            "created_at": datetime.now().isoformat(),
                            **metadata
                        }
                        
                        # Update category index
                        if category not in self._categories:
                            self._categories[category] = []
                        self._categories[category].append(doc_id)
                        
                        added_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Could not process document: {e}")
                        continue
                
                # Update stats
                self._stats.total_vectors = len(self._vectors)
                self._stats.index_size_bytes = sum(
                    vector.nbytes for vector in self._vectors.values()
                )
                
                logger.info(f"✅ Added {added_count} documents to vector index")
                return added_count > 0
                
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    async def semantic_search(self, 
                            query: str, 
                            category: Optional[str] = None,
                            top_k: int = 5,
                            min_score: float = 0.7) -> List[VectorSearchResult]:
        """Perform semantic search in the vector index."""
        if not self._embedding_model:
            logger.warning("No embedding model available for search")
            return []
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            with self._lock:
                # Generate query embedding
                query_embedding = self._embedding_model.encode([query])[0]
                
                # Filter by category if specified
                candidate_ids = []
                if category and category in self._categories:
                    candidate_ids = self._categories[category]
                else:
                    candidate_ids = list(self._vectors.keys())
                
                if not candidate_ids:
                    return []
                
                # Calculate similarities
                similarities = []
                for doc_id in candidate_ids:
                    if doc_id in self._vectors:
                        vector = self._vectors[doc_id]
                        similarity = cosine_similarity(
                            query_embedding.reshape(1, -1),
                            vector.reshape(1, -1)
                        )[0][0]
                        
                        if similarity >= min_score:
                            similarities.append((doc_id, similarity))
                
                # Sort by similarity and get top_k
                similarities.sort(key=lambda x: x[1], reverse=True)
                top_results = similarities[:top_k]
                
                # Create result objects
                results = []
                for doc_id, score in top_results:
                    if doc_id in self._metadata:
                        metadata = self._metadata[doc_id]
                        result = VectorSearchResult(
                            id=doc_id,
                            score=float(score),
                            content=metadata["content"],
                            metadata=metadata,
                            category=metadata.get("category")
                        )
                        results.append(result)
                
                # Update query stats
                query_time = (asyncio.get_event_loop().time() - start_time) * 1000
                self._update_query_stats(query_time)
                
                logger.debug(f"🔍 Semantic search found {len(results)} results for: {query[:50]}...")
                return results
                
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    async def hybrid_search(self, 
                          query: str,
                          categories: List[str] = None,
                          top_k_per_category: int = 3) -> Dict[str, List[VectorSearchResult]]:
        """Perform hybrid search across multiple categories."""
        results = {}
        
        if categories:
            for category in categories:
                category_results = await self.semantic_search(
                    query, category=category, top_k=top_k_per_category
                )
                if category_results:
                    results[category] = category_results
        else:
            # Search all categories
            for category in self._categories.keys():
                category_results = await self.semantic_search(
                    query, category=category, top_k=top_k_per_category
                )
                if category_results:
                    results[category] = category_results
        
        return results
    
    async def get_similar_content(self, 
                                content: str, 
                                exclude_id: Optional[str] = None,
                                top_k: int = 3) -> List[VectorSearchResult]:
        """Get similar content to the provided content."""
        if not self._embedding_model:
            return []
        
        try:
            with self._lock:
                content_embedding = self._embedding_model.encode([content])[0]
                
                similarities = []
                for doc_id, vector in self._vectors.items():
                    if exclude_id and doc_id == exclude_id:
                        continue
                    
                    similarity = cosine_similarity(
                        content_embedding.reshape(1, -1),
                        vector.reshape(1, -1)
                    )[0][0]
                    
                    similarities.append((doc_id, similarity))
                
                # Sort and get top results
                similarities.sort(key=lambda x: x[1], reverse=True)
                top_results = similarities[:top_k]
                
                results = []
                for doc_id, score in top_results:
                    if doc_id in self._metadata:
                        metadata = self._metadata[doc_id]
                        result = VectorSearchResult(
                            id=doc_id,
                            score=float(score),
                            content=metadata["content"],
                            metadata=metadata,
                            category=metadata.get("category")
                        )
                        results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"Error finding similar content: {e}")
            return []
    
    def _generate_document_id(self, content: str, category: str) -> str:
        """Generate unique document ID."""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return f"{category}_{content_hash}"
    
    def _update_query_stats(self, query_time_ms: float):
        """Update query performance statistics."""
        self._stats.total_queries += 1
        
        # Moving average
        if self._stats.avg_query_time_ms == 0:
            self._stats.avg_query_time_ms = query_time_ms
        else:
            self._stats.avg_query_time_ms = (
                self._stats.avg_query_time_ms * 0.9 + query_time_ms * 0.1
            )
    
    async def get_engine_stats(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics."""
        with self._lock:
            category_stats = {}
            for category, doc_ids in self._categories.items():
                category_stats[category] = len(doc_ids)
            
            stats = {
                "index_stats": {
                    "total_vectors": self._stats.total_vectors,
                    "index_size_bytes": self._stats.index_size_bytes,
                    "dimension": self._stats.dimension,
                    "max_vectors": self.max_vectors,
                    "utilization_percent": (self._stats.total_vectors / self.max_vectors) * 100
                },
                "performance": {
                    "avg_query_time_ms": self._stats.avg_query_time_ms,
                    "total_queries": self._stats.total_queries
                },
                "categories": category_stats,
                "health": {
                    "status": "healthy" if self._initialized else "not_initialized",
                    "embedding_model_available": self._embedding_model is not None
                }
            }
            
            return stats
    
    async def clear_index(self, category: Optional[str] = None):
        """Clear vectors from index."""
        with self._lock:
            if category and category in self._categories:
                # Clear specific category
                doc_ids_to_remove = self._categories[category]
                for doc_id in doc_ids_to_remove:
                    if doc_id in self._vectors:
                        del self._vectors[doc_id]
                    if doc_id in self._metadata:
                        del self._metadata[doc_id]
                del self._categories[category]
                
                logger.info(f"🗑️ Cleared {len(doc_ids_to_remove)} vectors from category: {category}")
            else:
                # Clear all
                self._vectors.clear()
                self._metadata.clear()
                self._categories.clear()
                logger.info("🗑️ Cleared all vectors from index")
            
            # Update stats
            self._stats.total_vectors = len(self._vectors)
            self._stats.index_size_bytes = sum(
                vector.nbytes for vector in self._vectors.values()
            )
    
    async def close(self):
        """Close the vector engine."""
        try:
            with self._lock:
                self._vectors.clear()
                self._metadata.clear()
                self._categories.clear()
            logger.info("✅ Enhanced Memory Vector Engine closed")
        except Exception as e:
            logger.warning(f"⚠️ Error during vector engine cleanup: {e}")
            # Continue cleanup even if there's an error

# Alias for compatibility
EnhancedPineconeEngine = EnhancedMemoryVectorEngine

async def create_pinecone_engine(api_key: Optional[str] = None,
                               index_name: str = "career-assistant-knowledge") -> Optional[EnhancedMemoryVectorEngine]:
    """Create enhanced memory vector engine instance."""
    logger.info("🚀 Creating Enhanced Memory Vector Engine...")
    
    engine = EnhancedMemoryVectorEngine()
    success = await engine.initialize()
    
    if success:
        logger.info("✅ Enhanced Memory Vector Engine created successfully")
        return engine
    else:
        logger.error("❌ Enhanced Memory Vector Engine initialization failed")
        return None

async def prepare_knowledge_base_for_pinecone(knowledge_files: List[str]) -> List[Dict[str, Any]]:
    """Prepare knowledge base files for vector indexing."""
    documents = []
    
    for file_path in knowledge_files:
        try:
            if not os.path.exists(file_path):
                logger.warning(f"Knowledge file not found: {file_path}")
                continue
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Determine category from filename
            filename = os.path.basename(file_path)
            if 'profile' in filename.lower():
                category = 'profile'
            elif 'experience' in filename.lower() or 'project' in filename.lower():
                category = 'experience'
            elif 'skill' in filename.lower() or 'tech' in filename.lower():
                category = 'skills'
            elif 'education' in filename.lower():
                category = 'education'
            else:
                category = 'general'
            
            # Split content into chunks
            chunks = _split_content_into_chunks(content)
            
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    documents.append({
                        "content": chunk.strip(),
                        "category": category,
                        "metadata": {
                            "source_file": file_path,
                            "chunk_index": i,
                            "total_chunks": len(chunks)
                        }
                    })
            
        except Exception as e:
            logger.error(f"Error processing knowledge file {file_path}: {e}")
    
    logger.info(f"✅ Prepared {len(documents)} document chunks from {len(knowledge_files)} files")
    return documents

def _split_content_into_chunks(content: str, max_chunk_size: int = 1000) -> List[str]:
    """Split content into manageable chunks."""
    # Simple paragraph-based splitting
    paragraphs = content.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = paragraph
        else:
            current_chunk += "\n\n" + paragraph if current_chunk else paragraph
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks 