# foundations/rag_engine.py
import logging
from pathlib import Path
import asyncio
import os
from typing import Optional
import faiss
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.prompts import PromptTemplate
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.faiss import FaissVectorStore

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- LlamaIndex Global Settings ---
# This sets up the models and configuration that will be used throughout the engine.
Settings.llm = Gemini(model_name="gemini-1.5-pro-latest")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.chunk_size = 512  # Smaller chunks for faster processing
Settings.chunk_overlap = 100  # Reduce overlap for speed

async def build_knowledge_index(knowledge_base_path: Path, index_path: Path):
    """
    Builds or loads a knowledge index from markdown files using FAISS.
    If an index exists at the specified path, it loads it. Otherwise, it builds
    a new one from the documents and saves it.
    """
    if not knowledge_base_path.exists() or not knowledge_base_path.is_dir():
        logger.error(f"Knowledge base directory not found: {knowledge_base_path}")
        return None

    try:
        # Check if a persisted index exists
        if index_path.exists() and any(index_path.iterdir()):
            logger.info(f"🧠 Loading existing knowledge index from: {index_path}")
            vector_store = FaissVectorStore.from_persist_dir(str(index_path))
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store, persist_dir=str(index_path)
            )
            index = load_index_from_storage(storage_context=storage_context)
            logger.info("✅ Knowledge index loaded successfully.")
            return index

        logger.info("🧠 No existing index found. Building a new one...")
        # Load all .md documents from the specified directory.
        reader = SimpleDirectoryReader(input_dir=knowledge_base_path, required_exts=[".md"])
        documents = reader.load_data()

        if not documents:
            logger.warning("No documents were loaded from the knowledge base.")
            return None

        # Create a FAISS vector store.
        d = Settings.embed_model._model.get_sentence_embedding_dimension()
        faiss_index = faiss.IndexFlatL2(d)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create the index from the documents
        logger.info(f"Indexing {len(documents)} documents...")
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        
        # Persist the index to disk for future use
        logger.info(f"💾 Saving index to: {index_path}")
        index.storage_context.persist(persist_dir=str(index_path))
        
        logger.info("✅ Knowledge index has been built and saved successfully.")
        return index

    except Exception as e:
        logger.critical(f"💥 Failed to build or load knowledge index: {e}", exc_info=True)
        return None

async def query_knowledge_index(rag_engine: VectorStoreIndex, query_text: str):
    """
    Asynchronously queries the knowledge index to get a synthesized response.
    """
    if not rag_engine:
        logger.error("RAG engine is not initialized.")
        return "Sorry, the AI engine is not available at the moment."

    logger.info(f"Synthesizing response for query: '{query_text}'")
    try:
        # Define a more detailed prompt template
        qa_prompt_template = (
            "You are Venkatesh Narra, a skilled Full-Stack and AI/ML Engineer. You are talking to a recruiter or potential employer. "
            "Your persona is professional, confident, and highly knowledgeable. Always speak in the first person ('I', 'my', 'we').\n"
            "Here is some context about your skills, experience, and projects:\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Based on the context and your persona, provide a comprehensive and thoughtful answer to the following question:\n"
            "Question: {query_str}\n\n"
            "Answer: "
        )
        qa_template = PromptTemplate(qa_prompt_template)
        
        # Create a query engine with the custom prompt - optimized for speed
        query_engine = rag_engine.as_query_engine(
            text_qa_template=qa_template,
            similarity_top_k=2,  # Fewer results for faster processing
            response_mode="compact"  # More concise processing
        )
        
        # Perform the query
        response = await query_engine.aquery(query_text)
        
        logger.info(f"✅ RAG engine generated response.")
        return str(response)

    except Exception as e:
        logger.error(f"💥 Error during RAG query: {e}", exc_info=True)
        return "An error occurred while processing your query. Please try again."

async def generate_profile_summary(rag_engine: VectorStoreIndex) -> str:
    """
    Generates a full professional profile by querying the index.
    The function signature is corrected to align with the AI server's calling convention.
    """
    if not rag_engine:
        return "Error: AI engine not available to generate profile."

    logger.info("🤖 Generating full professional profile...")
    try:
        query_engine = rag_engine.as_query_engine()
        prompt = (
            "Generate a comprehensive, professional summary for a software developer named Venkatesh Narra. "
            "Synthesize all available information from the knowledge base to create a complete profile in Markdown format. "
            "Include sections for Summary, Top Skills, Projects, Experience, and Education. "
            "Use the first person ('I', 'my')."
        )
        response = await query_engine.aquery(prompt)
        return str(response)
    except Exception as e:
        logger.error(f"💥 Error during profile generation: {e}", exc_info=True)
        return "Error: Could not generate the profile."

async def create_llm_client(api_key: Optional[str] = None) -> Gemini:
    """Create a Gemini LLM client for use in the enhanced router."""
    effective_api_key = api_key or os.getenv("GEMINI_API_KEY")
    
    if not effective_api_key:
        logger.warning("No Gemini API key provided")
        raise ValueError("Gemini API key is required")
    
    return Gemini(
        model_name="gemini-1.5-flash",
        api_key=effective_api_key,
        temperature=0.3,
        max_tokens=1000
    )
