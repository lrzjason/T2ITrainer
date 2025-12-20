# main.py
import subprocess
import sys
import os
import signal
import time
import shutil
from threading import Thread
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver
from pathlib import Path
import urllib.request
import urllib.error
import urllib.parse


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
    """Serve the built frontend on specified port with API proxy support"""
    frontend_dist = Path("frontend/dist").resolve()
    
    if not frontend_dist.exists():
        print(f"Frontend distribution directory not found: {frontend_dist}")
        return False
    
    # Create a custom handler to serve index.html for all routes (SPA support)
    # and proxy API requests to the backend
    class SPARequestHandler(BaseHTTPRequestHandler):
        def _proxy_request(self, method='GET', body=None):
            # Proxy API requests to backend server
            if self.path.startswith('/api/') or self.path.startswith('/custom_templates/'):
                try:
                    # Forward request to backend server
                    backend_url = f"http://localhost:8000{self.path}"
                    
                    # Create request to backend
                    req = urllib.request.Request(backend_url, data=body, method=method)
                    
                    # Copy headers from original request
                    for key, value in self.headers.items():
                        # Skip hop-by-hop headers
                        if key.lower() not in ('connection', 'upgrade', 'proxy-authenticate', 'proxy-authorization', 'te', 'trailers', 'transfer-encoding'):
                            req.add_header(key, value)
                    
                    # Send request to backend
                    with urllib.request.urlopen(req) as response:
                        # Send response back to client
                        self.send_response(response.getcode())
                        
                        # Copy headers from backend response
                        for key, value in response.headers.items():
                            # Skip hop-by-hop headers
                            if key.lower() not in ('connection', 'transfer-encoding'):
                                self.send_header(key, value)
                        
                        self.end_headers()
                        
                        # Copy response body
                        self.wfile.write(response.read())
                        
                except urllib.error.HTTPError as e:
                    # Forward HTTP errors from backend
                    self.send_response(e.code)
                    # Copy headers if available
                    for key, value in e.headers.items():
                        if key.lower() not in ('connection', 'transfer-encoding'):
                            self.send_header(key, value)
                    self.end_headers()
                    if e.fp:
                        self.wfile.write(e.fp.read())
                        
                except Exception as e:
                    # Handle other errors
                    self.send_error(500, f"Proxy error: {str(e)}")
                
                return True
            return False
        
        def do_GET(self):
            # Handle API and custom templates requests
            if self._proxy_request('GET'):
                return
            
            # Serve static files for non-API requests
            # If the requested path doesn't have a file extension,
            # serve index.html (needed for React Router)
            if not '.' in self.path.split('/')[-1]:
                file_path = frontend_dist / "index.html"
            else:
                # Remove leading slash and construct file path
                file_path = frontend_dist / self.path.lstrip('/')
            
            # Security check - ensure path is within frontend_dist
            try:
                file_path.resolve().relative_to(frontend_dist.resolve())
            except ValueError:
                self.send_error(403, "Forbidden")
                return
            
            # Check if file exists
            if file_path.exists() and file_path.is_file():
                # Guess content type based on file extension
                content_type = "text/html"
                if file_path.suffix == ".css":
                    content_type = "text/css"
                elif file_path.suffix == ".js":
                    content_type = "application/javascript"
                elif file_path.suffix == ".json":
                    content_type = "application/json"
                elif file_path.suffix == ".png":
                    content_type = "image/png"
                elif file_path.suffix == ".jpg" or file_path.suffix == ".jpeg":
                    content_type = "image/jpeg"
                elif file_path.suffix == ".ico":
                    content_type = "image/x-icon"
                
                # Read and serve file
                try:
                    with open(file_path, 'rb') as f:
                        content = f.read()
                    
                    self.send_response(200)
                    self.send_header("Content-type", content_type)
                    self.end_headers()
                    self.wfile.write(content)
                except Exception as e:
                    self.send_error(500, f"Error reading file: {str(e)}")
            else:
                # File not found, serve index.html for SPA routing
                index_file = frontend_dist / "index.html"
                if index_file.exists():
                    with open(index_file, 'rb') as f:
                        content = f.read()
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    self.wfile.write(content)
                else:
                    self.send_error(404, "File not found")
        
        def do_POST(self):
            # Handle POST requests by reading the body and proxying
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length) if content_length > 0 else None
            if self._proxy_request('POST', post_data):
                return
            self.send_error(404, "Not found")
        
        def do_PUT(self):
            # Handle PUT requests by reading the body and proxying
            content_length = int(self.headers.get('Content-Length', 0))
            put_data = self.rfile.read(content_length) if content_length > 0 else None
            if self._proxy_request('PUT', put_data):
                return
            self.send_error(404, "Not found")
        
        def do_DELETE(self):
            # Handle DELETE requests
            if self._proxy_request('DELETE'):
                return
            self.send_error(404, "Not found")
    
    # Start the server
    try:
        httpd = socketserver.TCPServer(("0.0.0.0", port), SPARequestHandler)
        print(f"Frontend served on http://localhost:{port}")
        print("Backend API running on http://localhost:8000")
        print("Press Ctrl+C to stop both servers")
        httpd.serve_forever()
    except OSError as e:
        if e.errno == 10048:  # Address already in use
            print(f"Port {port} is already in use. Try a different port.")
        else:
            print(f"Error starting server: {e}")
        return False
    except KeyboardInterrupt:
        print("\nFrontend server stopped")
        httpd.server_close()
        return True


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