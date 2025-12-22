# main.py
import subprocess
import sys
import time
import socket
import json
import os
from pathlib import Path

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
    
    # Return default values if config file doesn't exist or can't be loaded
    return {
        "backend_port": 8000,
        "frontend_dev_port": 5173,
        "frontend_preview_port": 7860
    }


# Load port configuration at module level
PORT_CONFIG = load_port_config()
BACKEND_PORT = PORT_CONFIG["backend_port"]
FRONTEND_PREVIEW_PORT = PORT_CONFIG["frontend_preview_port"]

def get_frontend_version():
    """Get the current frontend version from package.json"""
    frontend_dir = Path("frontend")
    package_json_path = frontend_dir / "package.json"
    
    if not package_json_path.exists():
        return None
    
    try:
        with open(package_json_path, 'r') as f:
            package_data = json.load(f)
            return package_data.get("version", "0.0.0")
    except (json.JSONDecodeError, KeyError):
        return None


def get_last_built_version():
    """Get the last built frontend version from public directory"""
    frontend_dir = Path("frontend")
    version_file_path = frontend_dir / "public" / "frontend_version.txt"
    
    if not version_file_path.exists():
        return None
    
    try:
        with open(version_file_path, 'r') as f:
            return f.read().strip()
    except:
        return None


def save_built_version(version):
    """Save the built frontend version to public directory"""
    frontend_dir = Path("frontend")
    version_file_path = frontend_dir / "public" / "frontend_version.txt"
    
    try:
        with open(version_file_path, 'w') as f:
            f.write(version)
        return True
    except:
        return False


def build_frontend():
    """Build the frontend if dist directory doesn't exist, is empty, or version has changed"""
    frontend_dir = Path("frontend")
    dist_dir = frontend_dir / "dist"
    
    # Get current frontend version
    current_version = get_frontend_version()
    if current_version is None:
        print("Warning: Could not determine current frontend version")
        current_version = "0.0.0"
    
    # Get last built version
    last_built_version = get_last_built_version()
    
    # Check if rebuild is needed
    needs_rebuild = False
    
    # Rebuild if dist directory doesn't exist or is empty
    if not dist_dir.exists() or not any(dist_dir.iterdir()):
        print("Frontend build missing or empty, rebuilding...")
        needs_rebuild = True
    # Rebuild if version has changed
    elif last_built_version != current_version:
        print(f"Frontend version changed from {last_built_version} to {current_version}, rebuilding...")
        needs_rebuild = True
    
    if needs_rebuild:
        print(f"Building frontend (version {current_version})...")
        try:
            # Set environment variables for build process
            build_env = os.environ.copy()
            build_env['VITE_API_URL'] = f'http://127.0.0.1:{BACKEND_PORT}'
            build_env['VITE_WS_URL'] = f'ws://127.0.0.1:{BACKEND_PORT}/ws/training'
            
            # Change to frontend directory and run npm build
            # Try npm first, then npm.cmd on Windows
            try:
                result = subprocess.run(["npm", "run", "build"], cwd=frontend_dir, check=True, env=build_env)
            except FileNotFoundError:
                # Try npm.cmd on Windows
                result = subprocess.run(["npm.cmd", "run", "build"], cwd=frontend_dir, check=True, env=build_env)
            print("Frontend built successfully!")
            # Save the built version
            save_built_version(current_version)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Frontend build failed with error: {e}")
            return False
        except FileNotFoundError:
            print("npm not found. Please install Node.js and ensure it's in your PATH.")
            print("Note: If you're on Windows, you might need to restart your command prompt after installing Node.js.")
            return False
    else:
        print(f"Frontend build already exists (version {current_version}).")
        return True

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

def serve_frontend(port=FRONTEND_PREVIEW_PORT):
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
        print(f"Backend API configured at http://127.0.0.1:{BACKEND_PORT} (used during build)")
        print("Press Ctrl+C to stop servers")
        
        # Try npm first, then npm.cmd on Windows
        try:
            preview_process = subprocess.Popen(["npm", "run", "preview", "--", "--port", str(port)], 
                                          cwd=frontend_dir)
            # Wait for the process to complete (it will run until interrupted)
            preview_process.wait()
        except FileNotFoundError:
            # Try npm.cmd on Windows
            preview_process = subprocess.Popen(["npm.cmd", "run", "preview", "--", "--port", str(port)], 
                                          cwd=frontend_dir)
            # Wait for the process to complete (it will run until interrupted)
            preview_process.wait()
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
        if 'preview_process' in locals():
            preview_process.terminate()
            try:
                preview_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                preview_process.kill()
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
        serve_frontend(FRONTEND_PREVIEW_PORT)
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