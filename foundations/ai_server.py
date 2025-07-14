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

# Smart Router
from .smart_router import MultiAgentSmartRouter

# MCP Client
from .mcp_client import create_mcp_client, MCPClient
from .simple_mcp_client import create_simple_mcp_client

# Database and Email utilities
from .db_utils import log_interaction, fetch_all_interactions, initialize_db
from .email_utils import send_contact_email

# Configuration
from .config import GEMINI_API_KEY

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class CareerAssistantService(career_assistant_pb2_grpc.CareerAssistantServicer):
    """Enhanced gRPC service with MCP-integrated smart routing."""
    
    def __init__(self):
        self.router = None
        self.mcp_client = None
        self.service_ready = False
        self.initialization_lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize the smart router with MCP client."""
        async with self.initialization_lock:
            if self.router is None:
                try:
                    logger.info("🚀 Initializing MCP-Enhanced Smart Query Router...")
                    
                    # Try to create MCP client with fallback to simple client
                    try:
                        self.mcp_client = await asyncio.wait_for(
                            create_mcp_client("career-assistant"),
                            timeout=15.0  # Reduced timeout for first attempt
                        )
                        if self.mcp_client:
                            logger.info("✅ MCP client created successfully")
                        else:
                            logger.warning("⚠️ MCP client creation returned None, trying simple fallback")
                            self.mcp_client = await create_simple_mcp_client("career-assistant")
                            if self.mcp_client:
                                logger.info("✅ Simple MCP client created successfully as fallback")
                            else:
                                logger.warning("⚠️ Both MCP clients failed, router will work without tools")
                    except asyncio.TimeoutError:
                        logger.warning("⚠️ MCP client creation timed out, trying simple fallback")
                        try:
                            self.mcp_client = await create_simple_mcp_client("career-assistant")
                            if self.mcp_client:
                                logger.info("✅ Simple MCP client created successfully as fallback")
                            else:
                                logger.warning("⚠️ Simple MCP client also failed, router will work without tools")
                        except Exception as e2:
                            logger.warning(f"⚠️ Simple MCP client also failed: {e2}")
                            self.mcp_client = None
                    except Exception as e:
                        logger.warning(f"⚠️ MCP client creation failed: {e}, trying simple fallback")
                        try:
                            self.mcp_client = await create_simple_mcp_client("career-assistant")
                            if self.mcp_client:
                                logger.info("✅ Simple MCP client created successfully as fallback")
                            else:
                                logger.warning("⚠️ Simple MCP client also failed, router will work without tools")
                        except Exception as e2:
                            logger.warning(f"⚠️ Simple MCP client also failed: {e2}")
                            self.mcp_client = None
                    
                    # Initialize router with or without MCP
                    self.router = MultiAgentSmartRouter(mcp_client=self.mcp_client)
                    logger.info("✅ MCP-Enhanced Smart Query Router initialized successfully")
                    
                except Exception as e:
                    logger.error(f"❌ Router initialization failed: {e}")
                    # Fallback to basic router
                    self.router = MultiAgentSmartRouter(mcp_client=None)
                    logger.info("✅ Fallback router initialized without MCP tools")
                
                self.service_ready = True


    async def ProcessQuery(self, request, context):
        if self.router is None:
            logger.error("💥 Router is None - this should not happen after proper initialization!")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Router initialization failed. Please restart the service.")
            return career_assistant_pb2.QueryResponse()

        query_text = request.query
        logger.info(f"🚀 Smart Router received query: '{query_text}'")

        try:
            # Use smart router for fast, accurate responses
            response_text = await self.router.route_query(query_text)
            
            # Log the interaction
            try:
                log_interaction(query_text, response_text)
            except Exception as e:
                logger.warning(f"Failed to log interaction: {e}")
            
            return career_assistant_pb2.QueryResponse(response=response_text)
        except Exception as e:
            logger.error(f"💥 Error during query processing: {e}", exc_info=True)
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
            if self.router is None:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("The Smart Router is still initializing. Please try again in a moment.")
                return career_assistant_pb2.ProfileResponse()

            profile_query = "Generate a comprehensive professional profile summary about myself"
            
            response = await self.router.route_query(profile_query)
            
            return career_assistant_pb2.ProfileResponse(content=str(response))
        except Exception as e:
            logger.error(f"💥 Error generating profile: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"An internal error occurred: {e}")
            return career_assistant_pb2.ProfileResponse(content="Error generating profile.")

async def initialize_router(service: CareerAssistantService, health_servicer: health.HealthServicer):
    # Mark as serving IMMEDIATELY to pass health checks
    health_servicer.set("foundations.CareerAssistantService", health_pb2.HealthCheckResponse.SERVING)
    logger.info("✅ Backend marked as SERVING immediately.")
    
    try:
        logger.info("🚀 Initializing Smart Query Router...")
        
        # Create MCP client for tool support (optional)
        try:
            mcp_client = await create_mcp_client("career-assistant")
            if mcp_client:
                logger.info("✅ MCP client created for router tool support.")
            else:
                logger.warning("⚠️ MCP client creation returned None, trying simple fallback")
                mcp_client = await create_simple_mcp_client("career-assistant")
                if mcp_client:
                    logger.info("✅ Simple MCP client created for router tool support.")
                else:
                    logger.warning("⚠️ Both MCP clients failed, router will work without tools")
        except Exception as e:
            logger.warning(f"⚠️ MCP client creation failed: {e}, trying simple fallback")
            try:
                mcp_client = await create_simple_mcp_client("career-assistant")
                if mcp_client:
                    logger.info("✅ Simple MCP client created for router tool support.")
                else:
                    logger.warning("⚠️ Simple MCP client also failed, router will work without tools")
            except Exception as e2:
                logger.warning(f"⚠️ Simple MCP client also failed: {e2}")
                mcp_client = None
        
        # Initialize service with MCP client
        await service.initialize()
        
        logger.info("✅ Multi-Agent Smart Router initialized successfully.")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Smart Router: {e}")
        # Mark as NOT serving since we failed to initialize properly
        health_servicer.set("foundations.CareerAssistantService", health_pb2.HealthCheckResponse.NOT_SERVING)
        raise RuntimeError(f"Failed to initialize Smart Router: {e}")

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
    

    init_task = asyncio.create_task(initialize_router(service, health_servicer))

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
