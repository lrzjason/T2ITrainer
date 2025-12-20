// Standalone WebSocket connection test for T2ITrainer backend

const WebSocket = require('ws'); // You may need to install ws: npm install ws

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
const ws = new WebSocket(BACKEND_URL);

ws.on('open', function open() {
  console.log('✅ WebSocket connection established');
  console.log('Sending test configuration...');
  
  // Send test configuration after connection is established
  ws.send(JSON.stringify(TEST_CONFIG));
});

ws.on('message', function message(data) {
  try {
    const parsedData = JSON.parse(data);
    console.log('📥 Received message from server:');
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
    console.log('📥 Received raw message:', data.toString());
  }
});

ws.on('close', function close() {
  console.log('');
  console.log('❌ WebSocket connection closed');
});

ws.on('error', function error(err) {
  console.log('');
  console.error('❌ WebSocket error occurred:');
  console.error('   Error:', err.message);
  console.error('   Code:', err.code);
});

// Set timeout to close connection after 30 seconds if no completion
setTimeout(() => {
  if (ws.readyState === WebSocket.OPEN) {
    console.log('');
    console.log('⏱️  Test timeout reached (30 seconds), closing connection');
    ws.close();
  }
}, 30000);