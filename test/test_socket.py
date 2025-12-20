#!/usr/bin/env python3
# Standalone WebSocket connection test for T2ITrainer backend

import asyncio
import websockets
import json

# Test configuration
BACKEND_URL = 'ws://127.0.0.1:8000/ws/training'
TEST_CONFIG = {
    "script": "train_longcat.py",
    "config_path": "test_config.json",
    "output_dir": "./test_output",
    "save_name": "test_run",
    "pretrained_model_name_or_path": "./test_model",
    "dataset_configs": [
        {
            "train_data_dir": "./test_data",
            "resolution": "512",
            "repeats": 1
        }
    ]
}

async def test_websocket_connection():
    print("WebSocket Connection Test")
    print("=======================")
    print(f"Attempting to connect to: {BACKEND_URL}")
    print("")
    
    try:
        # Establish WebSocket connection
        async with websockets.connect(BACKEND_URL) as websocket:
            print("[SUCCESS] WebSocket connection established")
            print("Sending test configuration...")

            # Send test configuration
            await websocket.send(json.dumps(TEST_CONFIG))

            print("Message sent, waiting for responses...")
            print("")

            # Listen for messages
            while True:
                try:
                    # Receive message with timeout
                    message = await asyncio.wait_for(websocket.recv(), timeout=30.0)

                    try:
                        parsed_data = json.loads(message)
                        print("[RECEIVED] Received message from server:")
                        print(f"   Type: {parsed_data.get('type', 'unknown')}")

                        if parsed_data.get('type') == 'output':
                            print(f"   Data: {parsed_data.get('data', '')}")
                        elif parsed_data.get('type') == 'complete':
                            print(f"   Status: {parsed_data.get('status', 'unknown')}")
                            print(f"   Return Code: {parsed_data.get('return_code', 'N/A')}")
                            print(f"   Message: {parsed_data.get('message', 'N/A')}")
                        elif parsed_data.get('type') == 'error':
                            print(f"   Error Message: {parsed_data.get('message', 'N/A')}")
                        elif parsed_data.get('type') == 'connection':
                            print(f"   Connection Status: {parsed_data.get('status', 'unknown')}")
                            print(f"   Message: {parsed_data.get('message', 'N/A')}")
                        else:
                            print(f"   Full message: {json.dumps(parsed_data, indent=2)}")
                    except json.JSONDecodeError:
                        print(f"[RECEIVED] Received raw message: {message}")

                    print("")

                    # If it's a completion message, exit the loop
                    if parsed_data.get('type') == 'complete':
                        print("[SUCCESS] Training process completed, closing connection")
                        break
                        
                except asyncio.TimeoutError:
                    print("⏱️  No message received within 30 seconds, closing connection")
                    break
                    
    except websockets.exceptions.ConnectionClosed as e:
        print(f"[ERROR] WebSocket connection closed: {e}")
    except ConnectionRefusedError:
        print("[ERROR] Connection refused - backend server may not be running")
    except Exception as e:
        print(f"[ERROR] WebSocket connection failed: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(test_websocket_connection())
    except KeyboardInterrupt:
        print("\n[INFO] Test interrupted by user")