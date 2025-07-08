# foundations/ai_server.py
import asyncio
import logging
from pathlib import Path
import grpc
from concurrent import futures
import os
from dotenv import load_dotenv
import google.generativeai as genai
import re

# Import generated gRPC classes
from . import career_assistant_pb2
from . import career_assistant_pb2_grpc

# Import project-specific modules
from .rag_engine import build_knowledge_index, query_knowledge_index, generate_profile_summary
from .email_utils import send_contact_email
from . import db_utils

# --- Environment & AI setup ──────────────────────────────────────────────────
load_dotenv(override=True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logging.warning("GEMINI_API_KEY not found in environment variables.")

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CareerAssistantService(career_assistant_pb2_grpc.CareerAssistantServicer):
    """
    The gRPC service that provides AI-powered responses.
    """
    def __init__(self):
        # Correctly resolve path from project root
        self.knowledge_base_path = Path(__file__).parent.parent / "agent_knowledge"
        self.index_path = Path(__file__).parent / "rag_index"
        self.rag_engine = None # Will be initialized by serve()
        db_utils.initialize_db()

    async def _initialize_rag(self):
        """Initializes the RAG engine from all knowledge documents asynchronously."""
        try:
            logger.info("🧠 Initializing Unified RAG Knowledge Base...")
            self.index_path.mkdir(exist_ok=True)
            
            md_files = list(self.knowledge_base_path.glob("*.md"))
            if not md_files:
                logger.warning("⚠️ No markdown files found in the knowledge base directory.")
                return
            logger.info(f"📚 Found {len(md_files)} documents to index.")
            
            self.rag_engine = await build_knowledge_index(md_files, self.index_path)
            logger.info("✅ Unified RAG Knowledge Base ready!")
        except Exception as e:
            logger.critical(f"💥 Failed to initialize RAG index: {e}", exc_info=True)
            self.rag_engine = None

    async def ProcessQuery(self, request, context):
        """Processes a user's query using the RAG engine."""
        query_text = request.query
        logger.info(f"🔍 Processing query with RAG: '{query_text}'")
        
        if self.rag_engine is None:
            logger.error("RAG index is not available. Cannot process query.")
            return career_assistant_pb2.QueryResponse(response="Sorry, the AI engine is currently offline. Please try again later.")

        if not GEMINI_API_KEY:
             return career_assistant_pb2.QueryResponse(response="Sorry, the AI engine is not configured correctly (missing API key).")
            
        try:
            search_results = await asyncio.to_thread(
                query_knowledge_index, self.rag_engine, self.index_path, query_text
            )
            
            if not search_results or not search_results.get('documents'):
                response_text = "I couldn't find specific information on that topic. Could you try rephrasing your question?"
                return career_assistant_pb2.QueryResponse(response=response_text)

            context_docs = "\n\n".join(doc[0] for doc in search_results['documents'])
            
            history_str = "\n".join([f"{msg.role}: {msg.content}" for msg in request.history])
            prompt = (
                "You are Venkatesh Narra, a Full-Stack Python Developer with expertise in AI/ML. You MUST speak in the first person using 'I', 'my', 'me'. Your persona is professional, confident, and detail-oriented.\n\n"
                "You must answer questions based ONLY on the information in the CONTEXT section and CONVERSATION HISTORY. Do not use any outside knowledge.\n\n"
                "### CRITICAL FORMATTING REQUIREMENTS ###\n"
                "You MUST format your responses with:\n"
                "1. **Use relevant emojis** (🔧, 💻, 🚀, 🏢, 📊, etc.) to make responses visually appealing\n"
                "2. **Bold headers** with `**Header:**` format\n"
                "3. **Use bullet points** (•, -, *) for lists\n"
                "4. **Code blocks** with ```python\n...\n```\n"
                "5. **Quotes** with `> `\n"
                "6. **Links** with `[text](url)`\n"
                "7. **Bold text** with `**text**`\n"
                "8. **Italic text** with `*text*`\n"
                "9. **Strikethrough text** with `~~text~~`\n"
                "10. **Inline code** with `` `code` ``\n\n"
                "### RESPONSE INSTRUCTIONS ###\n"
                "1. **Always format with emojis and structure** as shown above\n"
                "2. **Be comprehensive but concise** - include specific details from context\n"
                "3. **Use STAR method** for behavioral questions (Situation, Task, Action, Result)\n"
                "4. **Group technical skills** by category (Languages, Frameworks, Cloud, AI/ML)\n"
                "5. **Focus on recent work** at Veritis Group and AI projects when relevant\n"
                "6. **Include quantifiable achievements** (percentages, numbers, scale)\n"
                "7. **Sound confident and professional** while being authentic\n\n"
                "**🚨 CRITICAL: NO HALLUCINATION RULE 🚨**\n"
                "You MUST NEVER make up, assume, or infer information that is not explicitly stated in the CONTEXT.\n"
                "If the information to answer the question is NOT in the context, respond with: 'I don't have enough information on that specific topic. Please drop your email, and I'll get back to you with more details!'\n\n"
                f"CONVERSATION HISTORY:\n---------------------\n{history_str}\n---------------------\n\n"
                f"CONTEXT:\n---------------------\n{context_docs}\n---------------------\n\n"
                f"Question: {query_text}\n\n"
                "Answer (as Venkatesh Narra with proper formatting):"
            )

            model = genai.GenerativeModel("gemini-1.5-flash")
            gemini_response = await model.generate_content_async(prompt)
            
            response_text = gemini_response.text
            logger.info(f"✅ RAG generated response of {len(response_text)} characters.")
            
            db_utils.log_interaction(query_text, response_text)
            return career_assistant_pb2.QueryResponse(response=response_text)
            
        except Exception as e:
            logger.error(f"💥 Error during RAG query processing: {e}", exc_info=True)
            return career_assistant_pb2.QueryResponse(response="An error occurred while processing your request.")

    async def SubmitContactForm(self, request, context):
        """Handles contact form submissions by sending an email."""
        logger.info(f"📬 Received contact form submission from {request.name} <{request.email}>")
        try:
            success = await asyncio.to_thread(
                send_contact_email,
                name=request.name,
                email=request.email,
                message=request.message
            )
            if success:
                return career_assistant_pb2.ContactFormResponse(message="✅ Thank you! Your message has been sent. I'll get back to you shortly.")
            else:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details("Failed to send the email.")
                return career_assistant_pb2.ContactFormResponse(message="❌ Sorry, there was an error sending your message. Please try again later.")
        except Exception as e:
            logger.error(f"An unexpected error occurred in SubmitContactForm: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"An internal error occurred: {e}")
            return career_assistant_pb2.ContactFormResponse()

    async def GenerateProfile(self, request, context):
        """Generates a full professional profile summary."""
        logger.info("🤖 Received request to generate profile.")
        if not self.rag_engine:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("The AI engine is not ready. Please try again later.")
            return career_assistant_pb2.ProfileResponse()
        
        try:
            profile_content = await generate_profile_summary(self.rag_engine)
            return career_assistant_pb2.ProfileResponse(content=profile_content)
        except Exception as e:
            logger.error(f"💥 Error during profile generation: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Failed to generate profile.")
            return career_assistant_pb2.ProfileResponse()

    async def ScheduleMeeting(self, request, context):
        """Handles meeting scheduling requests."""
        logger.info(f"🗓️ Received meeting request from {request.email} for {request.time}")
        return career_assistant_pb2.MeetingResponse(
            success=True,
            message="Meeting request received! I will confirm the details and send a calendar invitation to your email shortly.",
            event_link="https://calendar.google.com"
        )

    async def GetAnalyticsData(self, request, context):
        """Fetches interaction data from the database."""
        logger.info("📊 Received request for analytics data.")
        try:
            interactions = await asyncio.to_thread(db_utils.fetch_all_interactions)
            response = career_assistant_pb2.AnalyticsResponse()
            for ix in interactions:
                response.interactions.add(
                    id=str(ix['id']),
                    query=ix['query'],
                    response=ix['response'],
                    timestamp=str(ix['timestamp'])
                )
            return response
        except Exception as e:
            logger.error(f"❌ Failed to fetch analytics data: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Failed to retrieve analytics data.")
            return career_assistant_pb2.AnalyticsResponse()

async def serve():
    """Starts the async gRPC server."""
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    service = CareerAssistantService()
    await service._initialize_rag()  # Ensure RAG is ready before serving
    
    career_assistant_pb2_grpc.add_CareerAssistantServicer_to_server(
        service, server
    )
    
    listen_addr = '[::]:50051'
    server.add_insecure_port(listen_addr)
    logger.info(f"🚀 Server starting on {listen_addr}")
    await server.start()
    await server.wait_for_termination()

if __name__ == '__main__':
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        logger.info("🔌 Server shutting down.")
    except Exception as e:
        logger.critical(f"💥 Server failed to start: {e}", exc_info=True) 