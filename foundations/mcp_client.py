"""
MCP Client Implementation - Windows-Optimized and Simplified
Connects to a stdio-based MCP server and provides a Pythonic interface.
"""
import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional
import subprocess

logger = logging.getLogger(__name__)

class SimpleMCPClient:
    """Simplified MCP client optimized for Windows with better error handling."""
    
    def __init__(self, process: subprocess.Popen, server_name: str):
        self.process = process
        self.server_name = server_name
        self.request_counter = 0
        self._initialized = False
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize the MCP client connection with better error handling."""
        async with self._lock:
            if self._initialized:
                return
                
            try:
                logger.info("🔄 Initializing MCP client connection...")
                
                # Send initialize request
                init_params = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "roots": {"listChanged": True},
                        "sampling": {}
                    },
                    "clientInfo": {
                        "name": "career-assistant-client",
                        "version": "1.0.0"
                    }
                }
                
                # Send the initialize request
                response = await self._send_request("initialize", init_params, timeout=8.0)
                
                if response and "result" in response:
                    logger.info("✅ MCP client initialized successfully")
                    self._initialized = True
                    return True
                else:
                    logger.error(f"❌ Initialize failed: {response}")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ MCP client initialization failed: {e}")
                return False
    
    async def _send_request(self, method: str, params: Dict[str, Any] = None, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
        """Send a JSON-RPC request to the MCP server with improved error handling."""
        if not self.process or self.process.poll() is not None:
            logger.error("❌ MCP server process is not running")
            return None
        
        self.request_counter += 1
        request_id = str(self.request_counter)
        
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method
        }
        
        if params:
            request["params"] = params
        
        try:
            # Send request
            request_json = json.dumps(request) + "\n"
            
            # Write to subprocess stdin
            self.process.stdin.write(request_json.encode('utf-8'))
            self.process.stdin.flush()
            
            # Read response with timeout
            try:
                response = await asyncio.wait_for(self._read_response(), timeout=timeout)
                
                if response and "error" in response:
                    logger.error(f"❌ RPC Error: {response['error']}")
                    return None
                
                return response
                
            except asyncio.TimeoutError:
                logger.error(f"❌ Request {method} timed out after {timeout}s")
                return None
                
        except Exception as e:
            logger.error(f"❌ Request {method} failed: {e}")
            return None
    
    async def _read_response(self) -> Optional[Dict[str, Any]]:
        """Read a single JSON-RPC response with better error handling."""
        try:
            # Read from subprocess stdout in async way
            while True:
                # Use run_in_executor to make blocking readline call async
                def read_line():
                    return self.process.stdout.readline()
                
                line = await asyncio.get_event_loop().run_in_executor(None, read_line)
                
                if not line:
                    logger.error("❌ MCP server closed stdout")
                    return None
                
                line_str = line.decode('utf-8').strip()
                if not line_str:
                    continue
                
                try:
                    response = json.loads(line_str)
                    return response
                except json.JSONDecodeError:
                    logger.debug(f"Skipping non-JSON line: {line_str}")
                    continue
                    
        except Exception as e:
            logger.error(f"❌ Error reading response: {e}")
            return None
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the MCP server."""
        try:
            if not self._initialized:
                success = await self.initialize()
                if not success:
                    logger.warning("⚠️ MCP client not initialized, returning empty tools list")
                    return []
                
            response = await self._send_request("tools/list", timeout=8.0)
            
            if response and "result" in response and "tools" in response["result"]:
                tools = response["result"]["tools"]
                logger.info(f"✅ Retrieved {len(tools)} tools from MCP server")
                return tools
            else:
                logger.warning("⚠️ No tools found in MCP server response")
                return []
                
        except Exception as e:
            logger.error(f"❌ Error listing tools: {e}")
            return []
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the MCP server."""
        try:
            if not self._initialized:
                success = await self.initialize()
                if not success:
                    return {"error": "MCP client not initialized"}
                
            params = {
                "name": name,
                "arguments": arguments
            }
            
            response = await self._send_request("tools/call", params, timeout=15.0)
            
            if response and "result" in response:
                result = response["result"]
                # Handle the MCP response format with content array
                if "content" in result and isinstance(result["content"], list):
                    # Extract text from content array
                    content_text = ""
                    for item in result["content"]:
                        if item.get("type") == "text":
                            content_text += item.get("text", "")
                    
                    try:
                        return {"result": json.loads(content_text)}
                    except json.JSONDecodeError:
                        return {"result": content_text}
                else:
                    return {"result": result}
            else:
                logger.warning(f"⚠️ Unexpected tools/call response: {response}")
                return {"error": "Invalid response format"}
                
        except Exception as e:
            logger.error(f"❌ Error calling tool {name}: {e}")
            return {"error": str(e)}

    def close(self):
        """Close the MCP client and terminate the server process."""
        try:
            if self.process and self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=5)
                logger.info("✅ MCP server process terminated")
        except Exception as e:
            logger.error(f"❌ Error closing MCP client: {e}")

async def start_mcp_server() -> Optional[subprocess.Popen]:
    """Start the MCP server subprocess with Windows-optimized settings."""
    try:
        command = [sys.executable, "-m", "foundations.mcp_server"]
        
        # Windows-specific subprocess settings
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
        
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            startupinfo=startupinfo,
            text=False,  # Use binary mode for better control
            bufsize=0    # Unbuffered
        )
        
        logger.info(f"🚀 Started MCP server subprocess with PID: {process.pid}")
        
        # Give the server a moment to start
        await asyncio.sleep(2.0)
        
        # Check if process is still running
        if process.poll() is not None:
            logger.error(f"❌ MCP server process exited with code {process.returncode}")
            return None
        
        return process
        
    except Exception as e:
        logger.error(f"❌ Failed to start MCP server: {e}")
        return None

async def create_mcp_client(server_name: str) -> Optional[SimpleMCPClient]:
    """Creates and initializes a simplified MCP client with better error handling."""
    logger.info(f"🚀 Creating MCP client for server: {server_name}")
    
    try:
        # Start the MCP server subprocess
        server_process = await start_mcp_server()
        
        if not server_process:
            logger.error("❌ Failed to start MCP server process")
            return None
        
        # Create the client
        client = SimpleMCPClient(server_process, server_name)
        
        # Try to initialize the connection
        success = await client.initialize()
        
        if success:
            logger.info("✅ MCP client created and initialized successfully")
            return client
        else:
            logger.error("❌ MCP client initialization failed")
            client.close()
            return None
        
    except Exception as e:
        logger.error(f"❌ Failed to create MCP client: {e}")
        return None

# Alias for backwards compatibility
MCPClient = SimpleMCPClient 