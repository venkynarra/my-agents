import asyncio
import logging
import grpc
from concurrent import futures
import os
from dotenv import load_dotenv
import json
import sys
from datetime import datetime

# gRPC imports
from . import career_assistant_pb2
from . import career_assistant_pb2_grpc
from google.protobuf import empty_pb2

# Health check imports
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

# LlamaIndex and Gemini
from llama_index.core.agent import ReActAgent
from llama_index.llms.gemini import Gemini
from llama_index.core.tools import FunctionTool

# MCP Client
from .mcp_client import create_mcp_client, MCPClient

# Database and Email utilities
from .db_utils import log_interaction, fetch_all_interactions, initialize_db
from .email_utils import send_contact_email

# Configuration
from .config import GEMINI_API_KEY

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def convert_mcp_tool_to_llama_tool(mcp_tool, mcp_client):
    """Convert an MCP tool to a LlamaIndex FunctionTool."""
    
    def create_async_function(tool):
        async def dynamic_func(**kwargs):
            logger.info(f"Calling MCP Tool '{tool['name']}' with args: {kwargs}")
            try:
                result = await mcp_client.call_tool(tool["name"], kwargs)
                
                # Handle the MCP response format
                if "content" in result and isinstance(result["content"], list):
                    # Extract text from content array
                    content_text = ""
                    for item in result["content"]:
                        if item.get("type") == "text":
                            content_text += item.get("text", "")
                    
                    # Try to parse as JSON if possible
                    try:
                        return json.loads(content_text)
                    except json.JSONDecodeError:
                        return {"result": content_text}
                else:
                    return result
                    
            except Exception as e:
                logger.error(f"Tool '{tool['name']}' error: {e}")
                return {"error": str(e)}

        dynamic_func.__name__ = tool["name"]
        dynamic_func.__doc__ = tool.get("description", "")
        return dynamic_func

    llama_tool = FunctionTool.from_defaults(
        fn=create_async_function(mcp_tool),
        name=mcp_tool["name"],
        description=mcp_tool.get("description", "")
    )
    
    return llama_tool

class CareerAssistantService(career_assistant_pb2_grpc.CareerAssistantServicer):
    def __init__(self):
        self.agent: ReActAgent | None = None
        # Initialize database
        try:
            initialize_db()
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")

    async def ProcessQuery(self, request, context):
        if self.agent is None:
            logger.error("💥 Agent is None - this should not happen after proper initialization!")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Agent initialization failed. Please restart the service.")
            return career_assistant_pb2.QueryResponse()

        query_text = request.query
        logger.info(f"🤖 Agent received query: '{query_text}'")

        try:
            # Enhanced prompt with knowledge base context
            instructed_query = (
                "You are Venkatesh Narra speaking directly. Respond as if you're having a natural conversation. "
                "Use 'I', 'my', 'me' throughout your responses. Be conversational and authentic. "
                "Here's my background: I'm a full-stack developer with 4+ years of experience. "
                "I currently work at Veritis Group Inc as a Software Development Engineer. Previously, I worked at TCS and Virtusa. "
                "I have a Master's in Computer Science from George Mason University (2022-2024) and a B.Tech from GITAM University (2018-2022). "
                "My expertise includes Python, Java, JavaScript, React, Node.js, AWS, Docker, Kubernetes, and AI/ML with TensorFlow and PyTorch. "
                "I've built AI-powered testing agents, clinical decision support tools, loan origination platforms, and various web applications. "
                "I'm passionate about solving complex problems and building scalable solutions. "
                f"Question: {query_text}"
            )
            response = await self.agent.achat(instructed_query)
            response_text = str(response)
            
            # Log the interaction
            try:
                log_interaction(query_text, response_text)
            except Exception as e:
                logger.warning(f"Failed to log interaction: {e}")
            
            return career_assistant_pb2.QueryResponse(response=response_text)
        except Exception as e:
            logger.error(f"💥 Error during agent query processing: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"An internal error occurred: {e}")
            return career_assistant_pb2.QueryResponse()

    async def SubmitContactForm(self, request, context):
        """Handle contact form submissions."""
        try:
            name = request.name
            email = request.email
            message = request.message
            
            logger.info(f"📧 Contact form submitted by {name} ({email})")
            
            # Send confirmation email
            email_sent = send_contact_email(name, email, message)
            
            if email_sent:
                return career_assistant_pb2.ContactFormResponse(
                    success=True,
                    message="Thank you for your message! I'll get back to you soon."
                )
            else:
                return career_assistant_pb2.ContactFormResponse(
                    success=False,
                    message="Your message was received, but there was an issue sending the confirmation email."
                )
        except Exception as e:
            logger.error(f"💥 Error processing contact form: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"An internal error occurred: {e}")
            return career_assistant_pb2.ContactFormResponse(
                success=False,
                message="An error occurred while processing your request."
            )

    async def ScheduleMeeting(self, request, context):
        """Handle meeting scheduling requests."""
        try:
            email = request.email
            time = request.time
            message = request.message
            
            logger.info(f"📅 Meeting request from {email} for {time}")
            
            # For now, we'll just return a success message with Calendly link
            # In a real implementation, you might integrate with a calendar API
            calendly_link = "https://calendly.com/venkateshnarra368"
            
            return career_assistant_pb2.MeetingResponse(
                success=True,
                message=f"Thank you for your meeting request! Please use the following link to schedule: {calendly_link}",
                event_link=calendly_link
            )
        except Exception as e:
            logger.error(f"💥 Error processing meeting request: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"An internal error occurred: {e}")
            return career_assistant_pb2.MeetingResponse(
                success=False,
                message="An error occurred while processing your meeting request.",
                event_link=""
            )

    async def GetAnalyticsData(self, request, context):
        """Retrieve analytics data from the database."""
        try:
            interactions = fetch_all_interactions()
            
            # Convert to protobuf format
            pb_interactions = []
            for interaction in interactions:
                pb_interaction = career_assistant_pb2.Interaction(
                    id=str(interaction['id']),
                    query=interaction['query'],
                    response=interaction['response'],
                    timestamp=interaction['timestamp']
                )
                pb_interactions.append(pb_interaction)
            
            return career_assistant_pb2.AnalyticsResponse(interactions=pb_interactions)
        except Exception as e:
            logger.error(f"💥 Error fetching analytics data: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"An internal error occurred: {e}")
            return career_assistant_pb2.AnalyticsResponse(interactions=[])

    async def GenerateProfile(self, request, context):
        """Generate a professional profile summary."""
        try:
            if self.agent is None:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("The AI agent is still initializing. Please try again in a moment.")
                return career_assistant_pb2.ProfileResponse()

            profile_query = (
                "Generate a comprehensive professional profile summary for Venkatesh Narra. "
                "Include key skills, experience, achievements, and what makes him unique as a "
                "Full-Stack Python Developer and AI/ML Engineer. Write in the first person."
            )
            
            response = await self.agent.achat(profile_query)
            
            return career_assistant_pb2.ProfileResponse(content=str(response))
        except Exception as e:
            logger.error(f"💥 Error generating profile: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"An internal error occurred: {e}")
            return career_assistant_pb2.ProfileResponse(content="Error generating profile.")

async def initialize_agent(service: CareerAssistantService, health_servicer: health.HealthServicer):
    # Mark as serving IMMEDIATELY to pass health checks
    health_servicer.set("foundations.CareerAssistantService", health_pb2.HealthCheckResponse.SERVING)
    logger.info("✅ Backend marked as SERVING immediately.")
    
    try:
        logger.info("🚀 Initializing Agent and its tools (MCP Client)...")
        
        # Create MCP client with proper timeout and error handling
        mcp_client = await create_mcp_client("career-assistant")
        
        # Get MCP tools
        mcp_tools = await mcp_client.list_tools()
        logger.info(f"✅ MCP client created, found {len(mcp_tools)} tools.")
        
        # Convert MCP tools to LlamaIndex tools
        llama_tools = []
        for mcp_tool in mcp_tools:
            try:
                llama_tool = await convert_mcp_tool_to_llama_tool(mcp_tool, mcp_client)
                llama_tools.append(llama_tool)
                logger.info(f"✅ Converted MCP tool: {mcp_tool['name']}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to convert MCP tool {mcp_tool['name']}: {e}")
        
        # Create the agent with MCP tools
        llm = Gemini(model_name="gemini-1.5-pro-latest", api_key=GEMINI_API_KEY)
        service.agent = ReActAgent.from_tools(llama_tools, llm=llm, verbose=True)
        
        logger.info(f"✅ Agent initialized successfully with {len(llama_tools)} tools.")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize MCP agent: {e}")
        # Mark as NOT serving since we failed to initialize properly
        health_servicer.set("foundations.CareerAssistantService", health_pb2.HealthCheckResponse.NOT_SERVING)
        raise RuntimeError(f"Failed to initialize MCP agent: {e}")

async def main():
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    service = CareerAssistantService()
    health_servicer = health.HealthServicer()

    career_assistant_pb2_grpc.add_CareerAssistantServicer_to_server(service, server)
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    server.add_insecure_port('[::]:50051')
    await server.start()
    logger.info("🚀 gRPC server started and listening on [::]:50051. Agent is initializing...")

    # Basic readiness check (server running)
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    

    init_task = asyncio.create_task(initialize_agent(service, health_servicer))

    try:
        await server.wait_for_termination()
    finally:
        logger.info("🔌 Shutting down server...")
        init_task.cancel()

if __name__ == '__main__':
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🔌 Server shutdown requested by user.")
    except Exception as e:
        logger.critical(f"💥 Main execution failed: {e}", exc_info=True)
