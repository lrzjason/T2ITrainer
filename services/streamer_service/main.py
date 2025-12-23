"""
WebSocket Streamer Service - Always-online WebSocket server for real-time output streaming.

This service:
- Maintains persistent WebSocket connections with clients
- Subscribes to job output channels via message queue (Redis Pub/Sub)
- Streams training output to connected clients in real-time
- Handles multiple concurrent connections per job
- Provides heartbeat/ping to keep connections alive

This service is completely decoupled from the training process - it only
subscribes to output messages from the message queue.
"""
import sys
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.shared.config import STREAMER_SERVICE_PORT, JOB_OUTPUT_PREFIX
from services.shared.message_queue import get_message_queue
from services.shared.sqlite_queue import get_sqlite_queue, OutputSubscriber

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="T2I Trainer WebSocket Streamer",
    version="2.0.0",
    description="WebSocket service for real-time training output streaming"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionManager:
    """Manages WebSocket connections and subscriptions"""
    
    def __init__(self):
        # job_id -> set of websocket connections
        self.active_connections: Dict[str, Set[WebSocket]] = defaultdict(set)
        # Global connections (not job-specific)
        self.global_connections: Set[WebSocket] = set()
        # Subscription tasks for each job
        self.subscription_tasks: Dict[str, asyncio.Task] = {}
        # SQLite subscribers for each job
        self.sqlite_subscribers: Dict[str, OutputSubscriber] = {}
        # Lock for thread-safe operations
        self.lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, job_id: str = None):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        
        async with self.lock:
            if job_id:
                self.active_connections[job_id].add(websocket)
                logger.info(f"Client connected for job {job_id}. Total: {len(self.active_connections[job_id])}")
                
                # Start subscription if this is the first connection for this job
                if job_id not in self.subscription_tasks or self.subscription_tasks[job_id].done():
                    self.subscription_tasks[job_id] = asyncio.create_task(
                        self._subscribe_to_job_output(job_id)
                    )
            else:
                self.global_connections.add(websocket)
                logger.info(f"Global client connected. Total: {len(self.global_connections)}")
    
    async def disconnect(self, websocket: WebSocket, job_id: str = None):
        """Handle WebSocket disconnection"""
        async with self.lock:
            if job_id:
                self.active_connections[job_id].discard(websocket)
                logger.info(f"Client disconnected for job {job_id}. Remaining: {len(self.active_connections[job_id])}")
                
                # Stop subscription if no more connections for this job
                if not self.active_connections[job_id]:
                    if job_id in self.subscription_tasks:
                        self.subscription_tasks[job_id].cancel()
                        del self.subscription_tasks[job_id]
                    del self.active_connections[job_id]
            else:
                self.global_connections.discard(websocket)
    
    async def send_to_job(self, job_id: str, message: dict):
        """Send a message to all connections for a specific job"""
        connections = self.active_connections.get(job_id, set()).copy()
        message_text = json.dumps(message)
        
        for connection in connections:
            try:
                await connection.send_text(message_text)
            except Exception as e:
                logger.error(f"Error sending to connection: {e}")
                # Remove dead connection
                await self.disconnect(connection, job_id)
    
    async def broadcast(self, message: dict):
        """Broadcast a message to all global connections"""
        message_text = json.dumps(message)
        
        for connection in self.global_connections.copy():
            try:
                await connection.send_text(message_text)
            except Exception as e:
                logger.error(f"Error broadcasting: {e}")
                await self.disconnect(connection)
    
    async def _subscribe_to_job_output(self, job_id: str):
        """Subscribe to job output using SQLite polling and forward messages to WebSocket clients"""
        sqlite_mq = get_sqlite_queue()
        subscriber = OutputSubscriber(job_id, sqlite_mq)
        self.sqlite_subscribers[job_id] = subscriber
        
        logger.info(f"Started SQLite subscription for job: {job_id}")
        
        try:
            while True:
                # Check if there are still active connections
                if not self.active_connections.get(job_id):
                    logger.info(f"No more connections for {job_id}, stopping subscription")
                    break
                
                # Get new messages from SQLite with shorter timeout for faster response
                messages = await subscriber.get_messages(timeout=0.05)
                
                for message in messages:
                    try:
                        await self.send_to_job(job_id, message)
                    except Exception as e:
                        logger.error(f"Error processing message for job {job_id}: {e}")
                        break
                
                # Smaller delay to prevent busy loop but allow faster updates
                await asyncio.sleep(0.005)
                
                # Check if job is complete
                if subscriber.is_complete:
                    logger.info(f"Job {job_id} completed, stopping subscription")
                    break
                
        except asyncio.CancelledError:
            logger.info(f"Subscription cancelled for {job_id}")
        except Exception as e:
            logger.error(f"Subscription error for {job_id}: {e}")
        finally:
            if job_id in self.sqlite_subscribers:
                del self.sqlite_subscribers[job_id]
            logger.info(f"Stopped SQLite subscription for job: {job_id}")


# Global connection manager
manager = ConnectionManager()


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info(f"WebSocket Streamer Service started on port {STREAMER_SERVICE_PORT}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    total_connections = sum(len(conns) for conns in manager.active_connections.values())
    total_connections += len(manager.global_connections)
    
    return {
        "status": "healthy",
        "service": "streamer_service",
        "active_connections": total_connections,
        "active_jobs": list(manager.active_connections.keys())
    }


@app.websocket("/ws/{job_id}")
async def websocket_job_endpoint(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for streaming output of a specific job.
    
    Connect to this endpoint after starting a training job to receive
    real-time output from the training process.
    """
    await manager.connect(websocket, job_id)
    
    # Send connection acknowledgment
    try:
        await websocket.send_text(json.dumps({
            "type": "connection",
            "status": "connected",
            "job_id": job_id,
            "message": f"Connected to job {job_id} output stream"
        }))
    except Exception as e:
        logger.error(f"Error sending connection ack: {e}")
        await manager.disconnect(websocket, job_id)
        return
    
    try:
        # Keep connection alive and handle incoming messages (if any)
        while True:
            try:
                # Wait for messages with timeout for heartbeat
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0  # Heartbeat interval
                )
                
                # Handle ping/pong or other client messages
                try:
                    message = json.loads(data)
                    if message.get("type") == "ping":
                        await websocket.send_text(json.dumps({
                            "type": "pong",
                            "timestamp": time.time()
                        }))
                except json.JSONDecodeError:
                    pass
                    
            except asyncio.TimeoutError:
                # Send heartbeat
                try:
                    await websocket.send_text(json.dumps({
                        "type": "heartbeat",
                        "timestamp": time.time()
                    }))
                except:
                    break
                    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for job {job_id}")
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
    finally:
        await manager.disconnect(websocket, job_id)


@app.websocket("/ws/training")
async def websocket_training_endpoint(websocket: WebSocket):
    """
    Legacy WebSocket endpoint for backward compatibility.
    
    This endpoint accepts configuration and directly runs training,
    streaming output back to the client in real-time.
    """
    await websocket.accept()
    
    process = None
    
    try:
        # Wait for configuration from client
        data = await websocket.receive_text()
        config_data = json.loads(data)
        
        # Extract job details
        script_path = config_data.get("script")
        config_path = config_data.get("config_path", "config.json")
        
        # Generate job ID
        from services.shared.utils import generate_job_id, save_config_to_file
        from services.shared.config import PROJECT_ROOT
        import subprocess
        import sys
        
        job_id = generate_job_id()
        
        # Save config
        saved_path = save_config_to_file(config_data)
        
        # Send acknowledgment with job ID
        await websocket.send_text(json.dumps({
            "type": "connection",
            "status": "connected",
            "job_id": job_id,
            "message": "WebSocket connected. Starting training..."
        }))
        
        logger.info(f"Starting training job {job_id}: {script_path}")
        
        # Validate script exists
        script_full_path = PROJECT_ROOT / script_path
        if not script_full_path.exists():
            await websocket.send_text(json.dumps({
                "type": "error",
                "status": "error",
                "message": f"Training script not found: {script_path}"
            }))
            return
        
        # Resolve config path
        import os
        if os.path.isabs(config_path):
            full_config_path = config_path
        else:
            full_config_path = str(PROJECT_ROOT / config_path)
        
        # Build command
        cmd = [sys.executable, str(script_full_path), "--config_path", full_config_path]
        
        # Set up environment for unbuffered output
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        # Send status update
        await websocket.send_text(json.dumps({
            "type": "status",
            "status": "running",
            "message": f"Starting: {' '.join(cmd)}"
        }))
        
        # Start subprocess
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            cwd=str(PROJECT_ROOT)
        )
        
        logger.info(f"Training process started with PID: {process.pid}")
        
        # Read output and stream to WebSocket
        async def read_output():
            while True:
                try:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    
                    line_str = line.decode('utf-8', errors='replace').rstrip('\n\r')
                    
                    if not line_str.strip():
                        continue
                    
                    # Check for training end marker
                    if "=========== End Training ===========" in line_str:
                        await websocket.send_text(json.dumps({
                            "type": "training_end",
                            "data": "Training completed successfully"
                        }))
                    
                    # Send output line
                    await websocket.send_text(json.dumps({
                        "type": "output",
                        "data": line_str.replace("\r", "\n")
                    }))
                    
                except Exception as e:
                    logger.error(f"Error reading output: {e}")
                    break
        
        # Create task for reading output
        read_task = asyncio.create_task(read_output())
        
        # Wait for either process completion or WebSocket disconnect
        try:
            while process.returncode is None:
                try:
                    # Check for client messages (with timeout for heartbeat)
                    msg = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                    # Handle ping messages
                    try:
                        msg_data = json.loads(msg)
                        if msg_data.get("type") == "ping":
                            await websocket.send_text(json.dumps({
                                "type": "pong",
                                "timestamp": time.time()
                            }))
                    except:
                        pass
                except asyncio.TimeoutError:
                    # Send heartbeat
                    await websocket.send_text(json.dumps({
                        "type": "heartbeat",
                        "timestamp": time.time()
                    }))
                
                # Check if process finished
                if process.returncode is not None:
                    break
                    
        except WebSocketDisconnect:
            logger.info(f"Client disconnected, terminating training process {process.pid}")
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                process.kill()
            return
        
        # Wait for output reading to complete
        await read_task
        
        # Wait for process to complete
        return_code = await process.wait()
        
        # Send completion message
        status = "success" if return_code == 0 else "error"
        await websocket.send_text(json.dumps({
            "type": "complete",
            "status": status,
            "return_code": return_code,
            "message": "Training completed successfully" if return_code == 0 
                      else f"Training failed with return code {return_code}"
        }))
        
        logger.info(f"Training job {job_id} completed with return code {return_code}")
        
    except WebSocketDisconnect:
        logger.info("Legacy WebSocket disconnected")
        if process and process.returncode is None:
            process.terminate()
    except Exception as e:
        logger.error(f"Legacy WebSocket error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "status": "error",
                "message": str(e)
            }))
        except:
            pass
    finally:
        if process and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except:
                process.kill()


@app.websocket("/ws/test")
async def websocket_test_endpoint(websocket: WebSocket):
    """Test WebSocket endpoint for connection verification"""
    await manager.connect(websocket)
    
    try:
        await websocket.send_text(json.dumps({
            "type": "connection",
            "status": "connected",
            "message": "Test WebSocket connected"
        }))
        
        while True:
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                response = {
                    "type": "echo",
                    "data": f"Echo: {message.get('message', data)}",
                    "timestamp": time.time()
                }
            except json.JSONDecodeError:
                response = {
                    "type": "echo",
                    "data": f"Echo: {data}",
                    "timestamp": time.time()
                }
            
            await websocket.send_text(json.dumps(response))
            
    except WebSocketDisconnect:
        logger.info("Test WebSocket disconnected")
    except Exception as e:
        logger.error(f"Test WebSocket error: {e}")
    finally:
        await manager.disconnect(websocket)


def main():
    """Run the WebSocket Streamer service"""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=STREAMER_SERVICE_PORT)


if __name__ == "__main__":
    main()
