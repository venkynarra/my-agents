"""
MCP Client Implementation
Connects to a stdio-based MCP server and provides a Pythonic interface.
"""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
import uuid
import re

from mcp.types import Tool

logger = logging.getLogger(__name__)

async def start_mcp_server() -> asyncio.subprocess.Process:
    """Start the MCP server subprocess."""
    command = ["python", "-m", "foundations.mcp_server"]
    
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    logger.info(f"🚀 Started MCP server subprocess with PID: {process.pid}")
    return process

async def wait_for_server_ready(process: asyncio.subprocess.Process, timeout: float = 60.0):
    """Wait for the MCP server to signal it's ready."""
    ready_signal = "MCP Server starting..."
    server_ready_event = asyncio.Event()
    
    async def watch_stderr():
        """Monitor stderr for the ready signal."""
        while True:
            try:
                line_bytes = await process.stderr.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode().strip()
                logger.warning(f"[MCP Server stderr] {line}")
                if ready_signal in line:
                    logger.info("✅ MCP Server has signaled it is ready.")
                    server_ready_event.set()
                    break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error watching MCP stderr: {e}")
                break
    
    # Start watching stderr
    watch_task = asyncio.create_task(watch_stderr())
    
    try:
        await asyncio.wait_for(server_ready_event.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        logger.error("❌ Timed out waiting for MCP server to become ready.")
        watch_task.cancel()
        process.terminate()
        await process.wait()
        raise RuntimeError("MCP server subprocess failed to start in time.")
    finally:
        watch_task.cancel()

class MCPClient:
    """A client for interacting with a stdio-based MCP server."""
    
    def __init__(self, process: asyncio.subprocess.Process, server_name: str):
        self.process = process
        self.server_name = server_name
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.request_id_counter = 0
        self._response_listener_task = None
        self._start_response_listener()
    
    def _start_response_listener(self):
        """Start the response listener task."""
        self._response_listener_task = asyncio.create_task(self._listen_for_responses())
    
    async def _listen_for_responses(self):
        """Listen for stdout and route responses to the correct future."""
        while True:
            try:
                # Read the JSON-RPC message
                line = await self.process.stdout.readline()
                if not line:
                    logger.warning("MCP client stdout stream closed.")
                    break
                
                try:
                    response = json.loads(line.decode('utf-8').strip())
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON response: {e}")
                    continue
                
                request_id = response.get("id")
                if request_id and request_id in self.pending_requests:
                    future = self.pending_requests.pop(request_id)
                    if "result" in response:
                        future.set_result(response)
                    elif "error" in response:
                        future.set_exception(Exception(f"RPC Error: {response['error']}"))
                    else:
                        future.set_result(response)
                
                logger.debug(f"<-- Received response: {response}")
                
            except asyncio.IncompleteReadError:
                logger.warning("MCP client stdout stream closed.")
                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in MCP response listener: {e}")
                break
    
    async def _send_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send a JSON-RPC request to the MCP server."""
        request_id = str(uuid.uuid4())
        
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method
        }
        
        # Only add params if they are provided and not empty
        if params:
            request["params"] = params
        
        future = asyncio.Future()
        self.pending_requests[request_id] = future
        
        try:
            request_json = json.dumps(request) + "\n"
            logger.debug(f"--> Sending request: {request}")
            
            self.process.stdin.write(request_json.encode('utf-8'))
            await self.process.stdin.drain()
            
            # Wait for the response
            response = await future
            return response
            
        except Exception as e:
            # Clean up the pending request
            self.pending_requests.pop(request_id, None)
            raise e
    
    async def _send_notification(self, method: str, params: Dict[str, Any] = None):
        """Send a JSON-RPC notification (no response expected)."""
        notification = {
            "jsonrpc": "2.0",
            "method": method
        }
        
        # Only add params if they are provided and not empty
        if params:
            notification["params"] = params
        
        notification_json = json.dumps(notification) + "\n"
        logger.debug(f"--> Sending notification: {notification}")
        
        self.process.stdin.write(notification_json.encode('utf-8'))
        await self.process.stdin.drain()
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the MCP server."""
        try:
            # The MCP specification uses "tools/list" method
            response = await self._send_request("tools/list")
            
            if "result" in response and "tools" in response["result"]:
                tools = response["result"]["tools"]
                logger.info(f"✅ Retrieved {len(tools)} tools from MCP server")
                return tools
            elif "result" in response:
                # Handle direct result format
                tools = response["result"]
                if isinstance(tools, list):
                    logger.info(f"✅ Retrieved {len(tools)} tools from MCP server")
                    return tools
            
            logger.warning(f"Unexpected response format for tools/list: {response}")
            return []
                
        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            return []
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the MCP server."""
        try:
            params = {
                "name": name,
                "arguments": arguments
            }
            
            response = await self._send_request("tools/call", params)
            
            if "result" in response:
                result = response["result"]
                # Handle the MCP response format with content array
                if "content" in result and isinstance(result["content"], list):
                    # Extract text from content array and try to parse as JSON
                    content_text = ""
                    for item in result["content"]:
                        if item.get("type") == "text":
                            content_text += item.get("text", "")
                    
                    try:
                        return json.loads(content_text)
                    except json.JSONDecodeError:
                        return {"result": content_text}
                else:
                    return result
            else:
                logger.warning(f"Unexpected response format for tools/call: {response}")
                return {"error": "Invalid response format"}
                
        except Exception as e:
            logger.error(f"Error calling tool {name}: {e}")
            return {"error": str(e)}

async def create_mcp_client(server_name: str) -> MCPClient:
    """Creates and initializes an MCP client."""
    logger.info(f"🚀 Creating MCP client for server: {server_name}")
    
    # Start the MCP server subprocess
    server_process = await start_mcp_server()
    
    # Wait for the server to be ready with a longer timeout
    await wait_for_server_ready(server_process, timeout=60)
    
    # Create the client
    client = MCPClient(server_process, server_name)
    
    # Initialize the connection with retries
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"🔄 Attempting to initialize MCP client (attempt {attempt + 1}/{max_retries})...")
            
            # Send initialize request with timeout
            init_params = {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {
                        "listChanged": True
                    },
                    "sampling": {}
                },
                "clientInfo": {
                    "name": "ai-agent-server",
                    "version": "1.0.0"
                }
            }
            
            # Send the initialize request
            response = await asyncio.wait_for(
                client._send_request("initialize", init_params),
                timeout=15.0
            )
            
            if response and "result" in response:
                logger.info("✅ MCP client initialized successfully.")
                
                # Send initialized notification - CRITICAL: This must be sent before any other requests
                await client._send_notification("notifications/initialized", {})
                logger.info("✅ MCP client initialization complete.")
                
                # Wait a moment for the server to process the initialized notification
                await asyncio.sleep(0.5)
                
                return client
            else:
                logger.warning(f"⚠️ Initialize response invalid: {response}")
                
        except asyncio.TimeoutError:
            logger.warning(f"⚠️ MCP initialize attempt {attempt + 1} timed out.")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
                continue
        except Exception as e:
            logger.warning(f"⚠️ MCP initialize attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
                continue
    
    # If we get here, all attempts failed
    logger.error("❌ Failed to initialize MCP client after all retries.")
    raise Exception("Failed to initialize MCP client") 