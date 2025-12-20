# main.py
import subprocess
import sys
import os
import signal
import time
import shutil
from threading import Thread
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver
from pathlib import Path


def build_frontend():
    """Build the frontend if dist directory doesn't exist or is empty"""
    frontend_dir = Path("frontend")
    dist_dir = frontend_dir / "dist"
    
    # Check if dist directory exists and has content
    if not dist_dir.exists() or not any(dist_dir.iterdir()):
        print("Building frontend...")
        try:
            # Change to frontend directory and run npm build
            result = subprocess.run(["npm", "run", "build"], cwd=frontend_dir, check=True, capture_output=True, text=True)
            print("Frontend built successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Frontend build failed with error: {e}")
            print(f"Error output: {e.stderr}")
            return False
    else:
        print("Frontend build already exists.")
        return True


def serve_frontend(port=7860):
    """Serve the built frontend on specified port"""
    frontend_dist = Path("frontend/dist")
    os.chdir(frontend_dist)
    
    handler = SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("0.0.0.0", port), handler)
    
    print(f"Frontend served on http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Frontend server stopped")
        httpd.server_close()


def run_backend():
    """Run the backend server"""
    try:
        subprocess.run([sys.executable, "backend_api.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Backend server failed with error: {e}")
    except KeyboardInterrupt:
        print("Backend server stopped")


def run_backend_async():
    """Run the backend server asynchronously"""
    try:
        backend_process = subprocess.Popen([sys.executable, "backend_api.py"])
        return backend_process
    except Exception as e:
        print(f"Failed to start backend server: {e}")
        return None


if __name__ == "__main__":
    print("Starting T2I Trainer application...")
    
    # Step 1: Build frontend if needed
    if not build_frontend():
        print("Exiting due to frontend build failure")
        sys.exit(1)
    
    # Step 2: Start backend server
    backend_process = run_backend_async()
    if backend_process is None:
        print("Failed to start backend server")
        sys.exit(1)
    
    # Give backend a moment to start
    time.sleep(3)
    
    # Step 3: Serve frontend
    print("Backend API running on http://localhost:8000")
    print("Press Ctrl+C to stop both servers")
    
    try:
        serve_frontend(7860)
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        if backend_process:
            backend_process.terminate()
            backend_process.wait()
        print("Servers stopped")