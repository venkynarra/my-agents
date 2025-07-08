# foundations/rag_engine.py
import asyncio
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from google.api_core.exceptions import PermissionDenied
from .config import GEMINI_API_KEY


# --- Environment & AI setup ──────────────────────────────────────────────────
load_dotenv(override=True)
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # This line is removed as per the new_code, as the LLM is now directly instantiated.
if GEMINI_API_KEY:
    # genai.configure(api_key=GEMINI_API_KEY) # This line is removed as per the new_code, as the LLM is now directly instantiated.
    pass # Keep this line to avoid breaking the new_code, but it's no longer needed.

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RAGEngine:
    def __init__(self, knowledge_base_dir: str):
        self.knowledge_base_dir = knowledge_base_dir
        self.vector_store: Optional[FAISS] = None
        self.chain: Optional[ConversationalRetrievalChain] = None
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
        self.embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        self._lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.documents: List[Document] = []
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    async def _load_and_process_documents(self):
        async with self._lock:
            if not self.vector_store:
                logger.info("Loading and processing documents for the first time.")
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(self.executor, self.load_documents)
                
                if self.vector_store:
                    logger.info("Vector store initialized.")
                    self.chain = ConversationalRetrievalChain.from_llm(
                        llm=self.llm,
                        retriever=self.vector_store.as_retriever(),
                        return_source_documents=True
                    )
                    logger.info("Conversational chain created.")
                else:
                    logger.error("Vector store could not be initialized. No documents were loaded or processed.")

    def load_documents(self):
        """Loads and chunks documents from a list of file paths."""
        logger.info(f"Building knowledge index from {len(self.knowledge_base_dir)} files...")
        all_chunks = []
        for file_path in self.knowledge_base_dir:
            try:
                # Assuming DirectoryLoader can handle both files and directories
                loader = DirectoryLoader(file_path, loader_cls=UnstructuredFileLoader)
                documents = loader.load()
                chunks = self.text_splitter.split_documents(documents)
                for chunk in chunks:
                    all_chunks.append(chunk)
                logger.info(f"📄 Processed {file_path}, created {len(chunks)} chunks.")
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
        
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
            response = self.llm.invoke(prompt)
            return response.content
        except PermissionDenied as e:
            logger.error(f"💥 Google API Permission Denied: {e}", exc_info=True)
            error_message = (
                "Error: The Generative Language API is not enabled for this project. "
                "Please enable it in your Google Cloud Console and try again."
            )
            return error_message
        except Exception as e:
            logger.error(f"💥 Error generating text with Gemini: {e}", exc_info=True)
            return "Error communicating with the generative model."


# --- Async Wrappers for Gradio ---
async def build_knowledge_index(file_paths: list, index_path: Path) -> RAGEngine:
    """Builds the RAG engine and knowledge index."""
    rag = RAGEngine(knowledge_base_dir=file_paths)
    # Run the synchronous document loading in a separate thread
    await rag._load_and_process_documents()
    return rag

async def query_knowledge_index(rag: RAGEngine, query: str, conversation_history: list) -> str:
    """Queries the RAG engine asynchronously."""
    return await asyncio.to_thread(rag.get_response, query, conversation_history)

async def generate_profile_summary(rag: RAGEngine) -> str:
    """Generates the profile summary asynchronously."""
    return await asyncio.to_thread(rag.generate_full_profile)
