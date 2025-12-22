#!/usr/bin/env python3
"""
Development server runner for T2I Trainer.
This script starts both the backend API server and the Vite development server with hot-reload capabilities.
"""
import os
import sys
import subprocess
import signal
import time
import socket
import json
from pathlib import Path

# Load port configuration
PORT_CONFIG_PATH = Path(__file__).parent / "port_config.json"


def load_port_config():
    """Load port configuration from JSON file"""
    if PORT_CONFIG_PATH.exists():
        try:
            with open(PORT_CONFIG_PATH, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load port config: {e}")
    
    # Return default values if config file doesn't exist or can't be loaded
    return {
        "backend_port": 8000,
        "frontend_dev_port": 5173,
        "frontend_preview_port": 7860
    }


# Load port configuration at module level
PORT_CONFIG = load_port_config()
BACKEND_PORT = PORT_CONFIG["backend_port"]
FRONTEND_DEV_PORT = PORT_CONFIG["frontend_dev_port"]


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print('\nShutting down development servers...')
    sys.exit(0)


def is_port_in_use(port):
    """Check if a port is already in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def start_backend():
    """Start the backend server if not already running"""
    if is_port_in_use(BACKEND_PORT):
        print(f"Backend server already running on port {BACKEND_PORT}")
        return None
    
    try:
        print(f"Starting backend server on port {BACKEND_PORT}...")
        backend_process = subprocess.Popen([sys.executable, "backend_api.py"])
        # Give backend time to start
        time.sleep(2)
        return backend_process
    except Exception as e:
        print(f"Failed to start backend server: {e}")
        return None


def main():
    """Main function to start both backend and frontend development servers"""
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    print("Starting T2I Trainer development environment...")
    
    # Step 1: Start backend server
    backend_process = start_backend()
    
    # Step 2: Start frontend development server
    frontend_dir = Path(__file__).parent / "frontend"
    
    if not frontend_dir.exists():
        print(f"Error: Frontend directory not found at {frontend_dir}")
        if backend_process:
            backend_process.terminate()
        sys.exit(1)
    
    print(f"Starting frontend development server in {frontend_dir}")
    
    # Check if npm is available
    npm_cmd = 'npm'
    try:
        result = subprocess.run(['npm', '--version'], 
                              capture_output=True, 
                              text=True, 
                              check=True,
                              cwd=frontend_dir,
                              encoding='utf-8',
                              errors='replace')
        print(f"Using npm version: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Try npm.cmd on Windows
        try:
            result = subprocess.run(['npm.cmd', '--version'], 
                                  capture_output=True, 
                                  text=True, 
                                  check=True,
                                  cwd=frontend_dir,
                                  encoding='utf-8',
                                  errors='replace')
            npm_cmd = 'npm.cmd'
            print(f"Using npm.cmd version: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: npm is not installed or not in PATH")
            if backend_process:
                backend_process.terminate()
            sys.exit(1)
    
    # Install dependencies if node_modules doesn't exist
    node_modules_path = frontend_dir / "node_modules"
    if not node_modules_path.exists():
        print("Installing dependencies...")
        try:
            subprocess.run([npm_cmd, 'install'], check=True, cwd=frontend_dir, encoding='utf-8', errors='replace')
            print("Dependencies installed successfully")
        except subprocess.CalledProcessError:
            print("Error: Failed to install dependencies")
            if backend_process:
                backend_process.terminate()
            sys.exit(1)
    
    # Start the development server
    print(f"Starting Vite development server on http://localhost:{FRONTEND_DEV_PORT}")
    print("Note: This server supports hot reload for development")
    print(f"Backend API is available at http://localhost:{BACKEND_PORT}")
    print("Press Ctrl+C to stop both servers")
    
    try:
        # Set environment variables for Vite dev server
        env = os.environ.copy()
        env['VITE_DEV_PORT'] = str(FRONTEND_DEV_PORT)
        env['VITE_API_TARGET'] = f'http://127.0.0.1:{BACKEND_PORT}'
        env['VITE_API_URL'] = f'http://127.0.0.1:{BACKEND_PORT}'
        env['VITE_WS_URL'] = f'ws://127.0.0.1:{BACKEND_PORT}/ws/training'
        
        # Run npm run dev
        dev_process = subprocess.Popen([npm_cmd, 'run', 'dev'], 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.STDOUT, 
                                     text=True,
                                     bufsize=1,
                                     universal_newlines=True,
                                     cwd=frontend_dir,
                                     encoding='utf-8',
                                     errors='replace',
                                     env=env)
        
        # Print output in real-time
        while True:
            output = dev_process.stdout.readline()
            if output == '' and dev_process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # Wait for process to complete
        dev_process.wait()
        
    except KeyboardInterrupt:
        print("\nReceived interrupt signal, shutting down...")
    except Exception as e:
        print(f"Error running development server: {e}")
    finally:
        # Cleanup backend if we started it
        if backend_process:
            backend_process.terminate()
            try:
                backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                backend_process.kill()
        sys.exit(0)


if __name__ == "__main__":
    main()