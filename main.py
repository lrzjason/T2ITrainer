# main.py
import subprocess
import sys
import os
import signal
import time
from threading import Thread

def run_backend():
    """Run the backend server"""
    try:
        subprocess.run([sys.executable, "backend_api.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Backend server failed with error: {e}")
    except KeyboardInterrupt:
        print("Backend server stopped")

def run_frontend():
    """Run the frontend development server"""
    try:
        # Change to frontend directory and run npm
        os.chdir("frontend")
        subprocess.run(["npm", "run", "dev"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Frontend server failed with error: {e}")
    except KeyboardInterrupt:
        print("Frontend server stopped")
    finally:
        os.chdir("..")

if __name__ == "__main__":
    print("Starting both servers...")
    print("Backend API will run on its default port")
    print("Frontend will run on its development port (usually 3000)")
    print("Press Ctrl+C to stop both servers")
    
    # Store process references for cleanup
    processes = []
    
    try:
        # Start backend
        backend_process = subprocess.Popen([sys.executable, "backend_api.py"])
        processes.append(backend_process)
        
        # Wait a moment for backend to initialize
        time.sleep(2)
        
        # Start frontend
        frontend_process = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd="frontend",
            shell=True  # Use shell for Windows compatibility
        )
        processes.append(frontend_process)
        
        # Wait for both processes
        for process in processes:
            process.wait()
            
    except KeyboardInterrupt:
        print("\nStopping both servers...")
        for process in processes:
            process.terminate()
        print("Servers stopped")