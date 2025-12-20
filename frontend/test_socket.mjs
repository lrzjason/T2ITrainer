// Standalone WebSocket connection test for T2ITrainer backend (ES Module version)

// For browser environments, we use the native WebSocket
// For Node.js with ES modules, we need to use dynamic import for ws
let WebSocketClient;

// Detect if we're in Node.js environment
if (typeof window === 'undefined') {
  // Node.js environment - need to dynamically import ws
  try {
    const wsModule = await import('ws');
    WebSocketClient = wsModule.default;
    console.log('Using ws library for Node.js');
  } catch (error) {
    console.error('ws library not found. Please run: npm install ws');
    process.exit(1);
  }
} else {
  // Browser environment - use native WebSocket
  WebSocketClient = WebSocket;
  console.log('Using native WebSocket for browser');
}

// Test configuration
const BACKEND_URL = 'ws://127.0.0.1:8000/ws/training';
const TEST_CONFIG = {
  script: "train_longcat.py",
  config_path: "test_config.json",
  output_dir: "./test_output",
  save_name: "test_run",
  pretrained_model_name_or_path: "./test_model",
  dataset_configs: [
    {
      train_data_dir: "./test_data",
      resolution: "512",
      repeats: 1
    }
  ]
};

console.log('WebSocket Connection Test');
console.log('=======================');
console.log(`Attempting to connect to: ${BACKEND_URL}`);
console.log('');

// Create WebSocket connection
let ws;
if (typeof window === 'undefined') {
  // Node.js environment
  ws = new WebSocketClient(BACKEND_URL);
} else {
  // Browser environment
  ws = new WebSocketClient(BACKEND_URL);
}

ws.onopen = function open() {
  console.log('[SUCCESS] WebSocket connection established');
  console.log('Sending test configuration...');
  
  // Send test configuration after connection is established
  const message = JSON.stringify(TEST_CONFIG);
  if (typeof window === 'undefined') {
    // Node.js
    ws.send(message);
  } else {
    // Browser
    ws.send(message);
  }
};

ws.onmessage = function message(event) {
  let data;
  if (typeof window === 'undefined') {
    // Node.js - data is a Buffer/Blob, convert to string
    data = event.toString();
  } else {
    // Browser - data is already a string
    data = event.data;
  }
  
  try {
    const parsedData = JSON.parse(data);
    console.log('[RECEIVED] Received message from server:');
    console.log('   Type:', parsedData.type);
    
    if (parsedData.type === 'output') {
      console.log('   Data:', parsedData.data);
    } else if (parsedData.type === 'complete') {
      console.log('   Status:', parsedData.status);
      console.log('   Return Code:', parsedData.return_code);
      console.log('   Message:', parsedData.message);
    } else if (parsedData.type === 'error') {
      console.log('   Error Message:', parsedData.message);
    } else if (parsedData.type === 'connection') {
      console.log('   Connection Status:', parsedData.status);
      console.log('   Message:', parsedData.message);
    } else {
      console.log('   Full message:', JSON.stringify(parsedData, null, 2));
    }
  } catch (e) {
    console.log('[RECEIVED] Received raw message:', data);
  }
};

ws.onclose = function close() {
  console.log('');
  console.log('[INFO] WebSocket connection closed');
};

ws.onerror = function error(err) {
  console.log('');
  console.error('[ERROR] WebSocket error occurred:');
  if (typeof window === 'undefined') {
    // Node.js error object
    console.error('   Error:', err.message);
    console.error('   Code:', err.code);
  } else {
    // Browser error event
    console.error('   Error:', err);
  }
};

// Set timeout to close connection after 30 seconds if no completion
setTimeout(() => {
  // Check readyState: 0=CONNECTING, 1=OPEN, 2=CLOSING, 3=CLOSED
  const readyState = typeof window === 'undefined' ? ws.readyState : ws.readyState;
  const readyStates = ['CONNECTING', 'OPEN', 'CLOSING', 'CLOSED'];
  
  if (readyState === 1) { // OPEN
    console.log('');
    console.log('[INFO] Test timeout reached (30 seconds), closing connection');
    ws.close();
  } else {
    console.log('[INFO] WebSocket is not open, current state:', readyStates[readyState] || 'UNKNOWN');
  }
}, 30000);