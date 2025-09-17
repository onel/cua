"""
FastAPI server for computer automation and control.

This module provides a WebSocket and HTTP API for controlling computer operations
including mouse/keyboard automation, file system operations, accessibility tree access,
and screen capture functionality.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, Header
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, Optional
import uvicorn
import logging
import asyncio
import json
import traceback
import inspect
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from .handlers.factory import HandlerFactory
import os
import aiohttp
import hashlib
import time

# Set up logging with more detail
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Configure WebSocket with larger message size
WEBSOCKET_MAX_SIZE = 1024 * 1024 * 10  # 10MB limit

# Configure application with WebSocket settings
app = FastAPI(
    title="Computer API",
    description="API for the Computer project",
    version="0.1.0",
    websocket_max_size=WEBSOCKET_MAX_SIZE,
)

protocol_version = 1
try:
    from importlib.metadata import version
    package_version = version("cua-computer-server")
except Exception:
    # Fallback for cases where package is not installed or importlib.metadata is not available
    try:
        import pkg_resources
        package_version = pkg_resources.get_distribution("cua-computer-server").version
    except Exception:
        package_version = "unknown"

accessibility_handler, automation_handler, diorama_handler, file_handler = HandlerFactory.create_handlers()
handlers = {
    "version": lambda: {"protocol": protocol_version, "package": package_version},
    # App-Use commands
    "diorama_cmd": diorama_handler.diorama_cmd,
    # Accessibility commands
    "get_accessibility_tree": accessibility_handler.get_accessibility_tree,
    "find_element": accessibility_handler.find_element,
    # Shell commands
    "run_command": automation_handler.run_command,
    # File system commands
    "file_exists": file_handler.file_exists,
    "directory_exists": file_handler.directory_exists,
    "list_dir": file_handler.list_dir,
    "read_text": file_handler.read_text,
    "write_text": file_handler.write_text,
    "read_bytes": file_handler.read_bytes,
    "write_bytes": file_handler.write_bytes,
    "get_file_size": file_handler.get_file_size,
    "delete_file": file_handler.delete_file,
    "create_dir": file_handler.create_dir,
    "delete_dir": file_handler.delete_dir,
    # Mouse commands
    "mouse_down": automation_handler.mouse_down,
    "mouse_up": automation_handler.mouse_up,
    "left_click": automation_handler.left_click,
    "right_click": automation_handler.right_click,
    "double_click": automation_handler.double_click,
    "move_cursor": automation_handler.move_cursor,
    "drag_to": automation_handler.drag_to,
    "drag": automation_handler.drag,
    # Keyboard commands
    "key_down": automation_handler.key_down,
    "key_up": automation_handler.key_up,
    "type_text": automation_handler.type_text,
    "press_key": automation_handler.press_key,
    "hotkey": automation_handler.hotkey,
    # Scrolling actions
    "scroll": automation_handler.scroll,
    "scroll_down": automation_handler.scroll_down,
    "scroll_up": automation_handler.scroll_up,
    # Screen actions
    "screenshot": automation_handler.screenshot,
    "get_cursor_position": automation_handler.get_cursor_position,
    "get_screen_size": automation_handler.get_screen_size,
    # Clipboard actions
    "copy_to_clipboard": automation_handler.copy_to_clipboard,
    "set_clipboard": automation_handler.set_clipboard,
}


class AuthenticationManager:
    """
    Manages authentication for container-based access control.
    
    Handles session caching and validation against the TryCUA API for
    container name and API key authentication.
    """
    
    def __init__(self):
        """Initialize the authentication manager with empty session cache."""
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.container_name = os.environ.get("CONTAINER_NAME")
    
    def _hash_credentials(self, container_name: str, api_key: str) -> str:
        """
        Create a hash of container name and API key for session identification.
        
        Args:
            container_name: The container name to hash
            api_key: The API key to hash
            
        Returns:
            str: SHA256 hash of the combined credentials
        """
        combined = f"{container_name}:{api_key}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _is_session_valid(self, session_data: Dict[str, Any]) -> bool:
        """
        Check if a session is still valid based on expiration time.
        
        Args:
            session_data: Dictionary containing session validity and expiration data
            
        Returns:
            bool: True if session is valid and not expired, False otherwise
        """
        if not session_data.get('valid', False):
            return False
        
        expires_at = session_data.get('expires_at', 0)
        return time.time() < expires_at
    
    async def auth(self, container_name: str, api_key: str) -> bool:
        """
        Authenticate container name and API key, using cached sessions when possible.
        
        Args:
            container_name: The container name to validate
            api_key: The API key to validate
            
        Returns:
            bool: True if authentication succeeds, False otherwise
        """
        # If no CONTAINER_NAME is set, always allow access (local development)
        if not self.container_name:
            logger.info("No CONTAINER_NAME set in environment. Allowing access (local development mode)")
            return True
        
        # Layer 1: VM Identity Verification
        if container_name != self.container_name:
            logger.warning(f"VM name mismatch. Expected: {self.container_name}, Got: {container_name}")
            return False
        
        # Create hash for session lookup
        session_hash = self._hash_credentials(container_name, api_key)
        
        # Check if we have a valid cached session
        if session_hash in self.sessions:
            session_data = self.sessions[session_hash]
            if self._is_session_valid(session_data):
                logger.info(f"Using cached authentication for container: {container_name}")
                return session_data['valid']
            else:
                # Remove expired session
                del self.sessions[session_hash]
        
        # No valid cached session, authenticate with API
        logger.info(f"Authenticating with TryCUA API for container: {container_name}")
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {api_key}"
                }
                
                async with session.get(
                    f"https://www.trycua.com/api/vm/auth?container_name={container_name}",
                    headers=headers,
                ) as resp:
                    is_valid = resp.status == 200 and bool((await resp.text()).strip())
                    
                    # Cache the result with 5 second expiration
                    self.sessions[session_hash] = {
                        'valid': is_valid,
                        'expires_at': time.time() + 5  # 5 seconds from now
                    }
                    
                    if is_valid:
                        logger.info(f"Authentication successful for container: {container_name}")
                    else:
                        logger.warning(f"Authentication failed for container: {container_name}. Status: {resp.status}")
                    
                    return is_valid
        
        except aiohttp.ClientError as e:
            logger.error(f"Failed to validate API key with TryCUA API: {str(e)}")
            # Cache failed result to avoid repeated requests
            self.sessions[session_hash] = {
                'valid': False,
                'expires_at': time.time() + 5
            }
            return False
        except Exception as e:
            logger.error(f"Unexpected error during authentication: {str(e)}")
            # Cache failed result to avoid repeated requests
            self.sessions[session_hash] = {
                'valid': False,
                'expires_at': time.time() + 5
            }
            return False


class ConnectionManager:
    """
    Manages WebSocket connections for the server.
    
    Handles connection lifecycle including accepting new connections
    and cleaning up disconnected clients.
    """
    
    def __init__(self):
        """Initialize the connection manager with an empty connection list."""
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """
        Accept a new WebSocket connection and add it to the active connections list.
        
        Args:
            websocket: The WebSocket connection to accept and manage
        """
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        """
        Remove a WebSocket connection from the active connections list.
        
        Args:
            websocket: The WebSocket connection to remove
        """
        self.active_connections.remove(websocket)


manager = ConnectionManager()
auth_manager = AuthenticationManager()


@app.websocket("/ws", name="websocket_endpoint")
async def websocket_endpoint(websocket: WebSocket):
    """
    Handle WebSocket connections for real-time command execution.
    
    Manages authentication for cloud deployments and processes incoming
    commands through the registered handlers. Supports both sync and async
    command handlers with parameter filtering.
    
    Args:
        websocket: The WebSocket connection to handle
    """
    global handlers

    # WebSocket message size is configured at the app or endpoint level, not on the instance
    await manager.connect(websocket)
    
    # Check if CONTAINER_NAME is set (indicating cloud provider)
    server_container_name = os.environ.get("CONTAINER_NAME")
    
    # If cloud provider, perform authentication handshake
    if server_container_name:
        try:
            logger.info(f"Cloud provider detected. CONTAINER_NAME: {server_container_name}. Waiting for authentication...")
            
            # Wait for authentication message
            auth_data = await websocket.receive_json()
            
            # Validate auth message format
            if auth_data.get("command") != "authenticate":
                await websocket.send_json({
                    "success": False,
                    "error": "First message must be authentication"
                })
                await websocket.close()
                manager.disconnect(websocket)
                return
            
            # Extract credentials
            client_api_key = auth_data.get("params", {}).get("api_key")
            client_container_name = auth_data.get("params", {}).get("container_name")
            
            # Validate credentials using AuthenticationManager
            if not client_api_key:
                await websocket.send_json({
                    "success": False,
                    "error": "API key required"
                })
                await websocket.close()
                manager.disconnect(websocket)
                return
            
            if not client_container_name:
                await websocket.send_json({
                    "success": False,
                    "error": "Container name required"
                })
                await websocket.close()
                manager.disconnect(websocket)
                return
            
            # Use AuthenticationManager for validation
            is_authenticated = await auth_manager.auth(client_container_name, client_api_key)
            if not is_authenticated:
                await websocket.send_json({
                    "success": False,
                    "error": "Authentication failed"
                })
                await websocket.close()
                manager.disconnect(websocket)
                return
            
            logger.info(f"Authentication successful for VM: {client_container_name}")
            await websocket.send_json({
                "success": True,
                "message": "Authentication successful"
            })
        
        except Exception as e:
            logger.error(f"Error during authentication handshake: {str(e)}")
            await websocket.send_json({
                "success": False,
                "error": "Authentication failed"
            })
            await websocket.close()
            manager.disconnect(websocket)
            return

    try:
        while True:
            try:
                data = await websocket.receive_json()
                command = data.get("command")
                params = data.get("params", {})

                if command not in handlers:
                    await websocket.send_json(
                        {"success": False, "error": f"Unknown command: {command}"}
                    )
                    continue

                try:
                    # Filter params to only include those accepted by the handler function
                    handler_func = handlers[command]
                    sig = inspect.signature(handler_func)
                    filtered_params = {k: v for k, v in params.items() if k in sig.parameters}
                    
                    # Handle both sync and async functions
                    if asyncio.iscoroutinefunction(handler_func):
                        result = await handler_func(**filtered_params)
                    else:
                        # Run sync functions in thread pool to avoid blocking event loop
                        result = await asyncio.to_thread(handler_func, **filtered_params)
                    await websocket.send_json({"success": True, **result})
                except Exception as cmd_error:
                    logger.error(f"Error executing command {command}: {str(cmd_error)}")
                    logger.error(traceback.format_exc())
                    await websocket.send_json({"success": False, "error": str(cmd_error)})

            except WebSocketDisconnect:
                raise
            except json.JSONDecodeError as json_err:
                logger.error(f"JSON decode error: {str(json_err)}")
                await websocket.send_json(
                    {"success": False, "error": f"Invalid JSON: {str(json_err)}"}
                )
            except Exception as loop_error:
                logger.error(f"Error in message loop: {str(loop_error)}")
                logger.error(traceback.format_exc())
                await websocket.send_json({"success": False, "error": str(loop_error)})

    except WebSocketDisconnect:
        logger.info("Client disconnected")
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Fatal error in websocket connection: {str(e)}")
        logger.error(traceback.format_exc())
        try:
            await websocket.close()
        except:
            pass
        manager.disconnect(websocket)


@app.post("/cmd")
async def cmd_endpoint(
    request: Request,
    container_name: Optional[str] = Header(None, alias="X-Container-Name"),
    api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """
    Backup endpoint for when WebSocket connections fail.
    
    Accepts commands via HTTP POST with streaming response and handles
    authentication for cloud deployments.
    
    Args:
        request: The HTTP request object containing the command data
        container_name: Container name from X-Container-Name header for cloud authentication
        api_key: API key from X-API-Key header for cloud authentication
        
    Returns:
        StreamingResponse: Streaming response containing command execution results
        
    Raises:
        HTTPException: For authentication failures, invalid requests, or unknown commands
    """
    global handlers
    
    # Parse request body
    try:
        body = await request.json()
        command = body.get("command")
        params = body.get("params", {})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {str(e)}")
    
    if not command:
        raise HTTPException(status_code=400, detail="Command is required")
    
    # Check if CONTAINER_NAME is set (indicating cloud provider)
    server_container_name = os.environ.get("CONTAINER_NAME")
    
    # If cloud provider, perform authentication
    if server_container_name:
        logger.info(f"Cloud provider detected. CONTAINER_NAME: {server_container_name}. Performing authentication...")
        
        # Validate required headers
        if not container_name:
            raise HTTPException(status_code=401, detail="Container name required")
        
        if not api_key:
            raise HTTPException(status_code=401, detail="API key required")
        
        # Validate with AuthenticationManager
        is_authenticated = await auth_manager.auth(container_name, api_key)
        if not is_authenticated:
            raise HTTPException(status_code=401, detail="Authentication failed")
    
    if command not in handlers:
        raise HTTPException(status_code=400, detail=f"Unknown command: {command}")
    
    async def generate_response():
        """
        Generate streaming response for the command execution.
        
        Yields:
            str: JSON-formatted response data as server-sent events
        """
        try:
            # Filter params to only include those accepted by the handler function
            handler_func = handlers[command]
            sig = inspect.signature(handler_func)
            filtered_params = {k: v for k, v in params.items() if k in sig.parameters}
            
            # Handle both sync and async functions
            if asyncio.iscoroutinefunction(handler_func):
                result = await handler_func(**filtered_params)
            else:
                # Run sync functions in thread pool to avoid blocking event loop
                result = await asyncio.to_thread(handler_func, **filtered_params)
            
            # Stream the successful result
            response_data = {"success": True, **result}
            yield f"data: {json.dumps(response_data)}\n\n"
            
        except Exception as cmd_error:
            logger.error(f"Error executing command {command}: {str(cmd_error)}")
            logger.error(traceback.format_exc())
            
            # Stream the error result
            error_data = {"success": False, "error": str(cmd_error)}
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, X-Container-Name, X-API-Key"
        }
    )


if __name__ == "__main__":
    """Run the FastAPI server when executed directly."""
    uvicorn.run(app, host="0.0.0.0", port=8000)
