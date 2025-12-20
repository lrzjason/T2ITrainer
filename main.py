# main.py
import subprocess
import sys
import time
import socket
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
            # Try npm first, then npm.cmd on Windows
            try:
                result = subprocess.run(["npm", "run", "build"], cwd=frontend_dir, check=True)
            except FileNotFoundError:
                # Try npm.cmd on Windows
                result = subprocess.run(["npm.cmd", "run", "build"], cwd=frontend_dir, check=True)
            print("Frontend built successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Frontend build failed with error: {e}")
            return False
        except FileNotFoundError:
            print("npm not found. Please install Node.js and ensure it's in your PATH.")
            print("Note: If you're on Windows, you might need to restart your command prompt after installing Node.js.")
            return False
    else:
        print("Frontend build already exists.")
        return True

def is_port_in_use(port):
    """Check if a port is already in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def start_backend():
    """Start the backend server if not already running"""
    if is_port_in_use(8000):
        print("Backend server already running on port 8000")
        return None
    
    try:
        print("Starting backend server...")
        backend_process = subprocess.Popen([sys.executable, "backend_api.py"])
        # Give backend time to start
        time.sleep(2)
        return backend_process
    except Exception as e:
        print(f"Failed to start backend server: {e}")
        return None

def serve_frontend(port=7860):
    """Serve the built frontend using npm run preview"""
    frontend_dir = Path("frontend").resolve()
    
    # Check if frontend directory exists
    if not frontend_dir.exists():
        print("ERROR: Frontend directory not found.")
        return False
    
    # Check if node_modules exists
    if not (frontend_dir / "node_modules").exists():
        print("ERROR: Node.js dependencies not found. Run 'npm install' in frontend directory first.")
        return False
    
    try:
        print(f"Starting frontend preview server on http://localhost:{port}")
        print("Press Ctrl+C to stop servers")
        
        # Try npm first, then npm.cmd on Windows
        try:
            result = subprocess.run(["npm", "run", "preview", "--", "--port", str(port)], 
                                  cwd=frontend_dir, check=True)
        except FileNotFoundError:
            # Try npm.cmd on Windows
            result = subprocess.run(["npm.cmd", "run", "preview", "--", "--port", str(port)], 
                                  cwd=frontend_dir, check=True)
        return True
    except FileNotFoundError:
        print("ERROR: npm not found. Please install Node.js and ensure it's in your PATH.")
        print("Note: If you're on Windows, you might need to restart your command prompt after installing Node.js.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Frontend preview server failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\nFrontend preview server stopped")
        return True

def main():
    print("Starting T2I Trainer application...")
    
    # Step 1: Build frontend if needed
    if not build_frontend():
        print("Exiting due to frontend build failure")
        sys.exit(1)
    
    # Step 2: Start backend server
    backend_process = start_backend()
    
    # Step 3: Serve frontend
    try:
        serve_frontend(7860)
    except KeyboardInterrupt:
        print("\nShutting down servers...")
    
    # Cleanup backend if we started it
    if backend_process:
        backend_process.terminate()
        try:
            backend_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            backend_process.kill()
            backend_process.wait()
    
    print("Servers stopped")

if __name__ == "__main__":
    main()