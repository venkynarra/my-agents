# foundations/rag_engine.py
import logging
from pathlib import Path
import os
from dotenv import load_dotenv
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.md import partition_md
import google.generativeai as genai
import asyncio
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

# --- Environment & AI setup ──────────────────────────────────────────────────
load_dotenv(override=True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UnifiedRAG:
    """A unified RAG engine that handles document loading, chunking, and querying."""
    def __init__(self, index_path: Path):
        self.index_path = index_path
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.vector_store = None

    def load_documents(self, file_paths: list[Path]):
        """Loads and chunks documents from a list of file paths."""
        logger.info(f"Building knowledge index from {len(file_paths)} files...")
        all_chunks = []
        for file in file_paths:
            try:
                elements = partition_md(filename=str(file))
                chunks = chunk_by_title(elements)
                for chunk in chunks:
                    all_chunks.append(Document(page_content=str(chunk), metadata={"source": file.name}))
                logger.info(f"📄 Processed {file.name}, created {len(chunks)} chunks.")
            except Exception as e:
                logger.error(f"Failed to process {file.name}: {e}")
        
        self.documents = all_chunks
        logger.info(f"✅ Knowledge index built with {len(self.documents)} total chunks.")
        
        # Create vector store from documents
        if self.documents:
            self.vector_store = FAISS.from_documents(self.documents, self.embedding_model)

    def get_retriever(self, k_results=5):
        """Builds and returns a retriever for the loaded documents."""
        if not self.vector_store:
            logger.error("Vector store not initialized. Cannot create retriever.")
            return None
        return self.vector_store.as_retriever(search_kwargs={"k": k_results})

    def get_response(self, query: str, conversation_history: list = None) -> str:
        """Gets a response from the RAG pipeline."""
        if not self.vector_store:
            return "Error: The knowledge base is not initialized."
            
        retriever = self.get_retriever()
        if not retriever:
            return "Error: Could not create a retriever."

        try:
            context_docs = retriever.get_relevant_documents(query)
            context = "\n\n---\n\n".join([doc.page_content for doc in context_docs])
            
            history_str = "\n".join([f"{msg.role}: {msg.content}" for msg in conversation_history]) if conversation_history else "No history."

            prompt = (
                "You are Venkatesh Narra, a Full-Stack Python Developer with expertise in AI/ML. You MUST speak in the first person using 'I', 'my', 'me'. Your persona is professional, confident, and detail-oriented.\n\n"
                "You must answer questions based ONLY on the information in the CONTEXT section and CONVERSATION HISTORY. Do not use any outside knowledge.\n\n"
                "### CRITICAL FORMATTING REQUIREMENTS ###\n"
                "You MUST format your responses with appropriate Markdown features like lists, bold text, and emojis (e.g., 🔧, 💻, 🚀, 🏢, 📊).\n\n"
                "**🚨 CRITICAL: NO HALLUCINATION RULE 🚨**\n"
                "If the information to answer the question is NOT in the context, respond with: 'I don't have enough information on that specific topic. Please drop your email, and I'll get back to you with more details!'\n\n"
                f"CONVERSATION HISTORY:\n---------------------\n{history_str}\n---------------------\n\n"
                f"CONTEXT:\n---------------------\n{context}\n---------------------\n\n"
                f"Question: {query}\n\n"
                "Answer (as Venkatesh Narra with proper formatting):"
            )
            return self.generate_text(prompt)
        except Exception as e:
            logger.error(f"💥 Error during RAG query processing: {e}", exc_info=True)
            return "An error occurred while processing your request."

    def generate_full_profile(self) -> str:
        """Generates a comprehensive profile by synthesizing all loaded documents."""
        if not self.documents:
            return "Error: No documents loaded to generate a profile from."
            
        try:
            logger.info("🤖 Generating full professional profile...")
            full_context = "\n\n---\n\n".join([doc.page_content for doc in self.documents])

            prompt = (
                "You are a professional profile writer. Your task is to generate a comprehensive and well-structured professional profile for Venkatesh Narra using the provided context. The profile should be formatted in Markdown and include the following sections:\n\n"
                "1.  **Contact Information:** List all links (LinkedIn, GitHub, LeetCode) and email.\n"
                "2.  **Summary:** A compelling professional summary.\n"
                "3.  **Top Skills:** A bulleted list of key technical skills.\n"
                "4.  **Cloud & DevOps Skills:** A separate bulleted list for Cloud/DevOps technologies.\n"
                "5.  **Projects:** A detailed look at 2-3 key projects, describing the tech stack and achievements for each.\n"
                "6.  **Experience:** A summary of professional roles with dates and key responsibilities.\n"
                "7.  **Education:** A summary of degrees and universities.\n\n"
                "### INSTRUCTIONS ###\n"
                "- Use clear headers with emojis (e.g., `## 🛠️ Top Skills`).\n"
                "- Extract and synthesize information from the entire context.\n"
                "- Do not invent information. Stick strictly to the provided text.\n"
                "- Ensure the formatting is clean, professional, and easy to read.\n\n"
                f"CONTEXT:\n---------------------\n{full_context}\n---------------------\n\n"
                "Generated Profile (in Markdown):"
            )
            return self.generate_text(prompt)
        except Exception as e:
            logger.error(f"💥 Error during full profile generation: {e}", exc_info=True)
            return "Error: Could not generate the profile."

    def generate_text(self, prompt: str) -> str:
        """Generates text using the Gemini model."""
        if not GEMINI_API_KEY:
            return "Error: Gemini API key not configured."
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"💥 Error generating text with Gemini: {e}", exc_info=True)
            return "Error communicating with the generative model."


# --- Async Wrappers for Gradio ---
async def build_knowledge_index(file_paths: list, index_path: Path) -> UnifiedRAG:
    """Builds the RAG engine and knowledge index."""
    rag = UnifiedRAG(index_path=index_path)
    # Run the synchronous document loading in a separate thread
    await asyncio.to_thread(rag.load_documents, file_paths)
    return rag

async def query_knowledge_index(rag: UnifiedRAG, query: str, conversation_history: list) -> str:
    """Queries the RAG engine asynchronously."""
    return await asyncio.to_thread(rag.get_response, query, conversation_history)

async def generate_profile_summary(rag: UnifiedRAG) -> str:
    """Generates the profile summary asynchronously."""
    return await asyncio.to_thread(rag.generate_full_profile)
