"""
Main launcher for T2I Trainer Services.

This script starts all three backend services:
1. API Service (FastAPI) - Port 8000
2. WebSocket Streamer Service - Port 8001
3. Worker Service - Background job processor

It also handles:
- Frontend building and serving
- Service health monitoring
- Graceful shutdown
"""
import subprocess
import sys
import time
import socket
import json
import os
import signal
import threading
from pathlib import Path
from typing import Optional, Dict, List

# Load port configuration
PORT_CONFIG_PATH = Path("port_config.json")


def load_port_config():
    """Load port configuration from JSON file"""
    if PORT_CONFIG_PATH.exists():
        try:
            with open(PORT_CONFIG_PATH, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load port config: {e}")
    
    return {
        "backend_port": 8000,
        "api_service_port": 8000,
        "streamer_service_port": 8001,
        "worker_service_port": 8002,
        "frontend_dev_port": 5173,
        "frontend_preview_port": 7860
    }


PORT_CONFIG = load_port_config()
API_PORT = PORT_CONFIG.get("api_service_port", 8000)
STREAMER_PORT = PORT_CONFIG.get("streamer_service_port", 8001)
FRONTEND_PREVIEW_PORT = PORT_CONFIG.get("frontend_preview_port", 7860)


class ServiceManager:
    """Manages all backend services"""
    
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.stop_event = threading.Event()
    
    def is_port_in_use(self, port: int) -> bool:
        """Check if a port is already in use"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    
    def start_service(self, name: str, module: str, port: Optional[int] = None) -> Optional[subprocess.Popen]:
        """Start a service as a subprocess"""
        if port and self.is_port_in_use(port):
            print(f"  {name}: Already running on port {port}")
            return None
        
        try:
            cmd = [sys.executable, "-m", module]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            self.processes[name] = process
            
            if port:
                print(f"  {name}: Started on port {port} (PID: {process.pid})")
            else:
                print(f"  {name}: Started (PID: {process.pid})")
            
            return process
            
        except Exception as e:
            print(f"  {name}: Failed to start - {e}")
            return None
    
    def start_api_service(self) -> Optional[subprocess.Popen]:
        """Start the API Service"""
        return self.start_service("API Service", "services.api_service.main", API_PORT)
    
    def start_streamer_service(self) -> Optional[subprocess.Popen]:
        """Start the WebSocket Streamer Service"""
        return self.start_service("Streamer Service", "services.streamer_service.main", STREAMER_PORT)
    
    def start_worker_service(self) -> Optional[subprocess.Popen]:
        """Start the Worker Service"""
        return self.start_service("Worker Service", "services.worker_service.main")
    
    def start_all_services(self):
        """Start all backend services"""
        print("\n[Starting Backend Services]")
        print("-" * 40)
        
        self.start_api_service()
        time.sleep(0.5)
        
        self.start_streamer_service()
        time.sleep(0.5)
        
        self.start_worker_service()
        
        print("-" * 40)
        print(f"\nServices Status:")
        print(f"  - API Service:      http://localhost:{API_PORT}")
        print(f"  - WebSocket:        ws://localhost:{STREAMER_PORT}/ws/{{job_id}}")
        print(f"  - Worker Service:   Running in background")
        print()
    
    def stop_all_services(self):
        """Stop all running services"""
        print("\n[Stopping Services]")
        
        for name, process in self.processes.items():
            if process and process.poll() is None:
                print(f"  Stopping {name}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"  Force killing {name}...")
                    process.kill()
                    process.wait()
        
        self.processes.clear()
        print("  All services stopped.")
    
    def monitor_services(self):
        """Monitor service health and restart if needed"""
        while not self.stop_event.is_set():
            time.sleep(10)
            
            for name, process in list(self.processes.items()):
                if process and process.poll() is not None:
                    return_code = process.returncode
                    print(f"\nWarning: {name} exited with code {return_code}")
                    
                    # Could add auto-restart logic here
                    # For now, just log
    
    def wait_for_services(self):
        """Wait for all services to complete"""
        try:
            while True:
                # Check if any service is still running
                running = [name for name, p in self.processes.items() 
                          if p and p.poll() is None]
                
                if not running:
                    break
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nReceived interrupt signal")


def get_frontend_version():
    """Get the current frontend version"""
    package_json = Path("frontend/package.json")
    if package_json.exists():
        try:
            with open(package_json, 'r') as f:
                return json.load(f).get("version", "0.0.0")
        except:
            pass
    return None


def get_last_built_version():
    """Get the last built version"""
    version_file = Path("frontend/public/frontend_version.txt")
    if version_file.exists():
        try:
            return version_file.read_text().strip()
        except:
            pass
    return None


def save_built_version(version: str):
    """Save the built version"""
    version_file = Path("frontend/public/frontend_version.txt")
    try:
        version_file.write_text(version)
    except:
        pass


def build_frontend():
    """Build the frontend if needed"""
    frontend_dir = Path("frontend")
    dist_dir = frontend_dir / "dist"
    
    current_version = get_frontend_version() or "0.0.0"
    last_version = get_last_built_version()
    
    needs_rebuild = False
    
    if not dist_dir.exists() or not any(dist_dir.iterdir()):
        print("Frontend build missing, rebuilding...")
        needs_rebuild = True
    elif last_version != current_version:
        print(f"Frontend version changed ({last_version} -> {current_version}), rebuilding...")
        needs_rebuild = True
    
    if needs_rebuild:
        print(f"Building frontend (version {current_version})...")
        try:
            env = os.environ.copy()
            env['VITE_API_URL'] = f'http://127.0.0.1:{API_PORT}'
            env['VITE_WS_URL'] = f'ws://127.0.0.1:{STREAMER_PORT}/ws'
            
            npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
            result = subprocess.run(
                [npm_cmd, "run", "build"],
                cwd=frontend_dir,
                check=True,
                env=env
            )
            
            save_built_version(current_version)
            print("Frontend built successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Frontend build failed: {e}")
            return False
        except FileNotFoundError:
            print("npm not found. Please install Node.js.")
            return False
    else:
        print(f"Frontend build exists (version {current_version})")
        return True


def serve_frontend(port: int = FRONTEND_PREVIEW_PORT):
    """Serve the frontend"""
    frontend_dir = Path("frontend").resolve()
    
    if not frontend_dir.exists():
        print("ERROR: Frontend directory not found.")
        return None
    
    if not (frontend_dir / "node_modules").exists():
        print("ERROR: Node modules not found. Run 'npm install' in frontend directory.")
        return None
    
    try:
        npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
        process = subprocess.Popen(
            [npm_cmd, "run", "preview", "--", "--port", str(port)],
            cwd=frontend_dir
        )
        
        print(f"Frontend serving at: http://localhost:{port}")
        return process
        
    except FileNotFoundError:
        print("ERROR: npm not found.")
        return None
    except Exception as e:
        print(f"ERROR: Failed to start frontend: {e}")
        return None


def main():
    """Main entry point"""
    print("=" * 50)
    print("T2I Trainer - Multi-Service Architecture")
    print("=" * 50)
    
    # Build frontend
    if not build_frontend():
        print("Warning: Frontend build failed, continuing with services only...")
    
    # Create service manager
    manager = ServiceManager()
    
    # Set up signal handlers
    def signal_handler(signum, frame):
        print("\nShutting down...")
        manager.stop_event.set()
        manager.stop_all_services()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start services
    manager.start_all_services()
    
    # Start frontend
    frontend_process = serve_frontend()
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=manager.monitor_services, daemon=True)
    monitor_thread.start()
    
    print("\nPress Ctrl+C to stop all services\n")
    
    try:
        # Wait for frontend process (blocking)
        if frontend_process:
            frontend_process.wait()
        else:
            # If no frontend, just wait for keyboard interrupt
            manager.wait_for_services()
            
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop_all_services()
        if frontend_process and frontend_process.poll() is None:
            frontend_process.terminate()
            frontend_process.wait()
    
    print("\nAll services stopped. Goodbye!")


if __name__ == "__main__":
    main()
