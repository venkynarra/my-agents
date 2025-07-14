"""
Enhanced AI Server with Async gRPC and Advanced Router Integration
Provides sub-2-second response times with enterprise-grade features.
"""
import asyncio
import logging
import grpc
from concurrent import futures
import os
from dotenv import load_dotenv
import time
from typing import Optional, Dict, Any

# gRPC imports
from . import career_assistant_pb2
from . import career_assistant_pb2_grpc
from google.protobuf import empty_pb2

# Health check imports
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

# Enhanced router
from .enhanced_smart_router import EnhancedMultiAgentRouter, create_enhanced_router

# Database and utilities
from .db_utils import log_interaction, fetch_all_interactions, initialize_db
from .email_utils import send_contact_email
from .config import GEMINI_API_KEY

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedCareerAssistantService(career_assistant_pb2_grpc.CareerAssistantServicer):
    """Enhanced gRPC service with next-generation multi-agent router."""
    
    def __init__(self):
        self.router: Optional[EnhancedMultiAgentRouter] = None
        self.service_ready = False
        self.initialization_lock = asyncio.Lock()
        self.performance_metrics = {
            "total_requests": 0,
            "sub_2s_responses": 0,
            "avg_response_time": 0.0,
            "cache_hits": 0
        }
        
    async def initialize(self):
        """Initialize the enhanced router with all components."""
        async with self.initialization_lock:
            if self.router is None:
                try:
                    logger.info("🚀 Initializing Enhanced Career Assistant Service...")
                    
                    # Get configuration from environment
                    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
                    pinecone_api_key = os.getenv("PINECONE_API_KEY")
                    gemini_api_key = os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
                    
                    # Create enhanced router with all components
                    self.router = await create_enhanced_router(
                        redis_url=redis_url,
                        pinecone_api_key=pinecone_api_key,
                        gemini_api_key=gemini_api_key
                    )
                    
                    if self.router:
                        logger.info("✅ Enhanced Career Assistant Service initialized successfully")
                        self.service_ready = True
                        return True
                    else:
                        logger.error("❌ Failed to create enhanced router")
                        return False
                        
                except Exception as e:
                    logger.error(f"❌ Service initialization failed: {e}")
                    return False
                
    async def ProcessQuery(self, request, context):
        """Process query using enhanced router with performance tracking."""
        if not self.service_ready:
            await self.initialize()
            
        if not self.router:
            logger.error("💥 Router is not available")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Service not properly initialized")
            return career_assistant_pb2.QueryResponse()

        query_text = request.query
        start_time = time.time()
        
        logger.info(f"🚀 Enhanced router processing query: '{query_text[:50]}...'")

        try:
            # Use enhanced router for optimal performance
            response_text = await self.router.route_query(query_text, use_cache=True)
            
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            
            # Update performance metrics
            self._update_performance_metrics(response_time_ms)
            
            # Log interaction (async to avoid blocking)
            asyncio.create_task(self._log_interaction_async(query_text, response_text))
            
            logger.info(f"✅ Enhanced response generated in {response_time_ms:.1f}ms")
            
            return career_assistant_pb2.QueryResponse(response=response_text)
            
        except Exception as e:
            logger.error(f"💥 Error during enhanced query processing: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal processing error: {e}")
            return career_assistant_pb2.QueryResponse()

    async def SubmitContactForm(self, request, context):
        """Handle contact form submissions with enhanced validation."""
        try:
            name = request.name
            email = request.email
            message = request.message
            
            # Enhanced validation
            if not all([name.strip(), email.strip(), message.strip()]):
                return career_assistant_pb2.ContactFormResponse(
                    success=False,
                    message="All fields are required."
                )
            
            # Email validation
            if "@" not in email or "." not in email:
                return career_assistant_pb2.ContactFormResponse(
                    success=False,
                    message="Please provide a valid email address."
                )
            
            # Send email asynchronously
            email_success = await asyncio.create_task(
                self._send_contact_email_async(name, email, message)
            )
            
            if email_success:
                return career_assistant_pb2.ContactFormResponse(
                    success=True,
                    message="Thank you for your message! I'll get back to you within 24 hours."
                )
            else:
                return career_assistant_pb2.ContactFormResponse(
                    success=False,
                    message="Sorry, there was an issue sending your message. Please try again."
                )
                
        except Exception as e:
            logger.error(f"Contact form error: {e}")
            return career_assistant_pb2.ContactFormResponse(
                success=False,
                message="An unexpected error occurred. Please try again."
            )

    async def ScheduleMeeting(self, request, context):
        """Handle meeting scheduling requests."""
        try:
            email = request.email
            time_slot = request.time
            message = request.message
            
            # Basic validation
            if not email or "@" not in email:
                return career_assistant_pb2.MeetingResponse(
                    success=False,
                    message="Please provide a valid email address.",
                    event_link=""
                )
            
            # For now, redirect to Calendly
            calendly_link = "https://calendly.com/venkateshnarra368"
            
            return career_assistant_pb2.MeetingResponse(
                success=True,
                message=f"Please use the provided link to schedule our meeting: {calendly_link}",
                event_link=calendly_link
            )
            
        except Exception as e:
            logger.error(f"Meeting scheduling error: {e}")
            return career_assistant_pb2.MeetingResponse(
                success=False,
                message="An error occurred while scheduling. Please try again.",
                event_link=""
            )

    async def GetAnalyticsData(self, request, context):
        """Get comprehensive analytics data from enhanced components."""
        try:
            if not self.router:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("Router not initialized")
                return career_assistant_pb2.AnalyticsResponse()

            # Get performance stats from enhanced router
            router_stats = await self.router.get_performance_stats()
            
            # Get recent interactions
            interactions = fetch_all_interactions()
            
            # Convert to protobuf format
            interaction_list = []
            for interaction in interactions[-50:]:  # Last 50 interactions
                interaction_proto = career_assistant_pb2.Interaction(
                    id=str(interaction[0]),
                    query=interaction[1],
                    response=interaction[2][:200] + "..." if len(interaction[2]) > 200 else interaction[2],
                    timestamp=interaction[3]
                )
                interaction_list.append(interaction_proto)
            
            return career_assistant_pb2.AnalyticsResponse(interactions=interaction_list)
            
        except Exception as e:
            logger.error(f"Analytics error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Analytics retrieval failed: {e}")
            return career_assistant_pb2.AnalyticsResponse()

    async def GenerateProfile(self, request, context):
        """Generate enhanced professional profile."""
        try:
            if not self.router:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("Router not initialized")
                return career_assistant_pb2.ProfileResponse()

            # Use enhanced router for profile generation
            profile_query = "Generate a comprehensive professional profile summary highlighting my background, skills, experience, and achievements"
            
            response = await self.router.route_query(profile_query, use_cache=True)
            
            return career_assistant_pb2.ProfileResponse(content=response)
            
        except Exception as e:
            logger.error(f"Profile generation error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Profile generation failed: {e}")
            return career_assistant_pb2.ProfileResponse(
                content="I'm Venkatesh Narra, a Software Development Engineer with 4+ years of experience in full-stack development and AI/ML integration."
            )

    async def GetPerformanceMetrics(self, request, context):
        """Get real-time performance metrics (custom endpoint)."""
        try:
            if not self.router:
                metrics = {"error": "Router not initialized"}
            else:
                # Get comprehensive metrics from enhanced router
                metrics = await self.router.get_performance_stats()
                
                # Add server-level metrics
                metrics["server_metrics"] = self.performance_metrics.copy()
                metrics["server_metrics"]["sub_2s_rate"] = (
                    self.performance_metrics["sub_2s_responses"] / 
                    max(self.performance_metrics["total_requests"], 1)
                ) * 100
            
            # Convert to JSON string for response
            import json
            metrics_json = json.dumps(metrics, indent=2)
            
            return career_assistant_pb2.QueryResponse(response=metrics_json)
            
        except Exception as e:
            logger.error(f"Performance metrics error: {e}")
            return career_assistant_pb2.QueryResponse(
                response=f"Error retrieving metrics: {e}"
            )

    async def HealthCheck(self, request, context):
        """Comprehensive health check of all components."""
        try:
            if not self.router:
                health_status = {
                    "status": "unhealthy",
                    "message": "Router not initialized",
                    "components": {}
                }
            else:
                health_data = await self.router.health_check()
                health_status = {
                    "status": "healthy" if health_data["router_initialized"] else "degraded",
                    "components": health_data["components"],
                    "message": "Enhanced AI Server operational"
                }
            
            import json
            return career_assistant_pb2.QueryResponse(response=json.dumps(health_status, indent=2))
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return career_assistant_pb2.QueryResponse(
                response=f"Health check failed: {e}"
            )

    def _update_performance_metrics(self, response_time_ms: float):
        """Update server performance metrics."""
        self.performance_metrics["total_requests"] += 1
        
        # Update average response time
        current_avg = self.performance_metrics["avg_response_time"]
        total = self.performance_metrics["total_requests"]
        self.performance_metrics["avg_response_time"] = (
            (current_avg * (total - 1)) + response_time_ms
        ) / total
        
        # Track sub-2s responses
        if response_time_ms < 2000:
            self.performance_metrics["sub_2s_responses"] += 1

    async def _log_interaction_async(self, query: str, response: str):
        """Asynchronously log interaction to database."""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, log_interaction, query, response)
        except Exception as e:
            logger.warning(f"Failed to log interaction: {e}")

    async def _send_contact_email_async(self, name: str, email: str, message: str) -> bool:
        """Asynchronously send contact email."""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, send_contact_email, name, email, message)
            return result
        except Exception as e:
            logger.error(f"Failed to send contact email: {e}")
            return False

    async def close(self):
        """Clean shutdown of the service."""
        if self.router:
            await self.router.close()
        logger.info("✅ Enhanced Career Assistant Service shutdown complete")


class EnhancedHealthServicer(health_pb2_grpc.HealthServicer):
    """Enhanced health check service."""
    
    def __init__(self, career_service):
        self.career_service = career_service

    async def Check(self, request, context):
        """Perform health check."""
        try:
            if self.career_service.service_ready:
                status = health_pb2.HealthCheckResponse.SERVING
            else:
                status = health_pb2.HealthCheckResponse.NOT_SERVING
                
            return health_pb2.HealthCheckResponse(status=status)
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return health_pb2.HealthCheckResponse(
                status=health_pb2.HealthCheckResponse.NOT_SERVING
            )


async def serve_enhanced():
    """Start the enhanced gRPC server."""
    # Initialize database
    initialize_db()
    
    # Create enhanced service
    career_service = EnhancedCareerAssistantService()
    
    # Initialize the service
    await career_service.initialize()
    
    # Create gRPC server with async support
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.keepalive_time_ms', 60000),
            ('grpc.keepalive_timeout_ms', 5000),
            ('grpc.keepalive_permit_without_calls', True),
            ('grpc.http2.max_pings_without_data', 0),
            ('grpc.http2.min_time_between_pings_ms', 10000),
            ('grpc.http2.min_ping_interval_without_data_ms', 300000)
        ]
    )
    
    # Add services
    career_assistant_pb2_grpc.add_CareerAssistantServicer_to_server(career_service, server)
    
    # Add health service
    health_servicer = EnhancedHealthServicer(career_service)
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    
    # Configure listen address
    listen_addr = '[::]:50051'
    server.add_insecure_port(listen_addr)
    
    logger.info(f"🚀 Enhanced Career Assistant gRPC Server starting on {listen_addr}")
    
    try:
        # Start server
        await server.start()
        logger.info("✅ Enhanced gRPC Server started successfully")
        
        # Wait for termination
        await server.wait_for_termination()
        
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down Enhanced gRPC Server...")
        await career_service.close()
        await server.stop(5)
        logger.info("✅ Enhanced gRPC Server shutdown complete")


def main():
    """Main entry point for enhanced server."""
    try:
        asyncio.run(serve_enhanced())
    except KeyboardInterrupt:
        logger.info("Enhanced server interrupted by user")
    except Exception as e:
        logger.error(f"Enhanced server error: {e}")


if __name__ == '__main__':
    main() 