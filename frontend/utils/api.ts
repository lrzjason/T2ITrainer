// API utility functions for T2I Trainer

// Determine the API base URL based on environment
const getApiBaseUrl = (): string => {
  // In production builds, check for environment variable
  // In development, vite will handle proxying if configured
  // For static serving via the proxy, use relative paths
  // If running from a different origin, use the configured backend URL
  const viteApiUrl = import.meta.env.VITE_API_URL;
  if (viteApiUrl) {
    return viteApiUrl;
  }

  // Check if we're running on the proxy server (port 3000) that should proxy API requests
  const currentPort = window.location.port;
  if (currentPort === '3000') {
    // When served from port 3000, relative paths should work as they'll be proxied
    return '';
  }

  // Default to the expected backend location
  return 'http://127.0.0.1:8000';
};

export const API_BASE_URL = getApiBaseUrl();

// Generic API call function
export const apiCall = async (endpoint: string, options: RequestInit = {}) => {
  // If API_BASE_URL is empty (for proxy server), use just the endpoint
  // Otherwise, use the full URL
  const url = API_BASE_URL ? `${API_BASE_URL}${endpoint}` : endpoint;

  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`API call failed: ${response.status} ${response.statusText}. Details: ${errorText}`);
  }

  return response.json();
};

// Training WebSocket connection for real-time output
// Get WebSocket URL from environment or use default
const getWebSocketUrl = (): string => {
  return getWebSocketUrlForEndpoint('training');
};

// Get WebSocket URL for a specific endpoint
const getWebSocketUrlForEndpoint = (endpoint: string): string => {
  // Check for environment variable first
  const wsUrl = import.meta.env.VITE_WS_URL;
  console.log('[WebSocket Debug] VITE_WS_URL from environment:', wsUrl);
  if (wsUrl) {
    // Modify the URL to point to the correct endpoint
    try {
      const urlObj = new URL(wsUrl);
      const baseUrl = `${urlObj.protocol}//${urlObj.host}`;
      console.log('[WebSocket Debug] Using VITE_WS_URL with endpoint:', `${baseUrl}/ws/${endpoint}`);
      return `${baseUrl}/ws/${endpoint}`;
    } catch (e) {
      // Fallback to regex if URL parsing fails
      const baseUrl = wsUrl.replace(/\/ws\/.*$/, '');
      console.log('[WebSocket Debug] Using VITE_WS_URL with endpoint (fallback):', `${baseUrl}/ws/${endpoint}`);
      return `${baseUrl}/ws/${endpoint}`;
    }
  }

  // Default WebSocket URL
  const apiBaseUrl = getApiBaseUrl();
  console.log('[WebSocket Debug] API base URL:', apiBaseUrl);
  
  // Even if API base URL is empty (proxy scenario), we should still connect to the backend WebSocket
  // Check if we're running on the proxy server (port 3000) and use default backend URL
  const currentPort = window.location.port;
  if ((!apiBaseUrl || apiBaseUrl === '') && currentPort === '3000') {
    console.log('[WebSocket Debug] Using default backend WebSocket URL for proxy scenario');
    return `ws://127.0.0.1:8000/ws/${endpoint}`;
  }
  
  if (!apiBaseUrl || apiBaseUrl === '') {
    // If API base URL is empty, likely using proxy - use relative path which will be handled by proxy
    // In this case, we need to construct the WebSocket URL based on the current page's protocol and host
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host || '127.0.0.1:3000'; // default to localhost if host is not available
    const fallbackUrl = `${protocol}//${host}/ws/${endpoint}`;
    console.log('[WebSocket Debug] Using fallback URL:', fallbackUrl);
    return fallbackUrl;
  } else if (apiBaseUrl.startsWith('http://')) {
    // Convert HTTP to WS
    try {
      const urlObj = new URL(apiBaseUrl);
      return `ws://${urlObj.host}/ws/${endpoint}`;
    } catch (e) {
      // If URL parsing fails, fallback to replace method
      return apiBaseUrl.replace('http://', 'ws://') + `/ws/${endpoint}`;
    }
  } else if (apiBaseUrl.startsWith('https://')) {
    // Convert HTTPS to WSS
    try {
      const urlObj = new URL(apiBaseUrl);
      return `wss://${urlObj.host}/ws/${endpoint}`;
    } catch (e) {
      // If URL parsing fails, fallback to replace method
      return apiBaseUrl.replace('https://', 'wss://') + `/ws/${endpoint}`;
    }
  } else {
    // If it's already just a host:port format
    return `ws://${apiBaseUrl}/ws/${endpoint}`;
  }
};

class TrainingWebSocketManager {
  private ws: WebSocket | null = null;
  private onOutputReceived: ((data: any) => void) | null = null;
  private debugEnabled = false;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectInterval = 3000; // 3 seconds
  private shouldReconnect = false; // Flag to control if reconnection should happen
  private connectionState: 'disconnected' | 'connecting' | 'connected' = 'disconnected'; // Track connection state

  connect = (onOutput: (data: any) => void, wsUrl?: string) => {
    // Use provided URL or get from environment/config
    const finalWsUrl = wsUrl || getWebSocketUrl();
    console.log('[WebSocket Debug] Final WebSocket URL:', finalWsUrl);

    // Close existing connection if any
    if (this.ws) {
      this.disconnect();
    }

    // Set the output handler
    this.onOutputReceived = onOutput;
    if (this.debugEnabled) console.log('[WebSocket Debug] Output receiver set');
    this.shouldReconnect = true; // Enable reconnection when establishing a new connection
    this.connectionState = 'connecting';
    this.reconnectAttempts = 0; // Reset reconnect attempts

    if (this.debugEnabled) console.log('[WebSocket Debug] Attempting to connect to:', finalWsUrl);

    try {
      console.log('[WebSocket Debug] Creating WebSocket connection to:', finalWsUrl);
      this.ws = new WebSocket(finalWsUrl);

      this.ws.onopen = (event) => {
        this.connectionState = 'connected';
        console.log('[WebSocket Debug] WebSocket connection opened successfully', event);
        this.reconnectAttempts = 0; // Reset reconnect attempts on successful connection
        
        // Send a connection established message to the frontend
        if (this.onOutputReceived) {
          this.onOutputReceived({
            type: 'connection',
            status: 'connected',
            message: 'WebSocket connection established'
          });
        }
      };

      this.ws.onmessage = (event) => {
        console.log('[WebSocket Debug] Received message:', event.data);
        try {
          // Try to parse as JSON first
          const data = JSON.parse(event.data);
          console.log('[WebSocket Debug] Parsed JSON data:', data);
          if (this.onOutputReceived) {
            // Add extra debugging for output messages
            if (data.type === 'output') {
              console.log('[WebSocket Debug] Forwarding output message to frontend:', data.data);
            }
            this.onOutputReceived(data);
          } else {
            console.log('[WebSocket Debug] No output receiver registered');
          }
        } catch (e) {
          // If JSON parsing fails, check if it's a plain string
          console.log('[WebSocket Debug] Failed to parse as JSON, treating as plain text');
          if (this.onOutputReceived) {
            // If it looks like a plain string, send it as output data
            if (typeof event.data === 'string' && event.data.trim().length > 0) {
              this.onOutputReceived({ type: 'output', data: event.data });
            } else {
              // Otherwise, send as raw data
              this.onOutputReceived({ type: 'raw', data: event.data });
            }
          }
        }
      };

      this.ws.onclose = (event) => {
        const prevState = this.connectionState;
        this.connectionState = 'disconnected';
        console.log('[WebSocket Debug] WebSocket connection closed:', {
          code: event.code,
          reason: event.reason,
          wasClean: event.wasClean
        });
        console.log('Training WebSocket connection closed');

        // Send a disconnection message to the frontend
        if (this.onOutputReceived) {
          this.onOutputReceived({
            type: 'connection',
            status: 'disconnected',
            message: `WebSocket connection closed (code: ${event.code})`
          });
        }

        // Only attempt to reconnect if we want to reconnect AND it wasn't a normal closure
        // Also only reconnect if we were previously connected (not failed connections)
        if (this.shouldReconnect && event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) { // 1000 = normal closure
          // Don't reconnect immediately after a failed initial connection
          if (prevState === 'connected' || this.reconnectAttempts > 0) {
            this.reconnectAttempts++;
            console.log(`[WebSocket Debug] Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
            setTimeout(() => {
              this.connect(onOutput, finalWsUrl);
            }, this.reconnectInterval);
          }
        }
      };

      this.ws.onerror = (error) => {
        // Note: Don't change connectionState to 'disconnected' on error because
        // the onclose event will be triggered after onerror for the same connection issue
        console.error('[WebSocket Debug] WebSocket error occurred:', error);
        console.error('Training WebSocket error:', error);
        
        // Send an error message to the frontend
        if (this.onOutputReceived) {
          this.onOutputReceived({
            type: 'error',
            status: 'error',
            message: `WebSocket error: ${error || 'Unknown error'}`
          });
        }
      };
    } catch (error) {
      console.error('[WebSocket Debug] Failed to create WebSocket connection:', error);
      if (this.debugEnabled) console.error('[WebSocket Debug] WebSocket creation failed:', error);
      this.connectionState = 'disconnected';
      this.ws = null;
      throw error;
    }

    // Return the WebSocket instance so the caller can set additional handlers
    return this.ws;
  };

  sendConfig = (config: any) => {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      console.log('[WebSocket Debug] Sending config:', config);
      this.ws.send(JSON.stringify(config));
      console.log('[WebSocket Debug] Config sent successfully');
    } else {
      const readyState = this.ws ? this.ws.readyState : 'NO_CONNECTION';
      console.log('[WebSocket Debug] WebSocket not connected. Ready state:', readyState);
      throw new Error(`Training WebSocket is not connected. Ready state: ${readyState}`);
    }
  };

  sendRawMessage = (message: any) => {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      console.log('[WebSocket Debug] Sending raw message:', message);
      this.ws.send(JSON.stringify(message));
      console.log('[WebSocket Debug] Raw message sent successfully');
    } else {
      const readyState = this.ws ? this.ws.readyState : 'NO_CONNECTION';
      console.log('[WebSocket Debug] WebSocket not connected. Ready state:', readyState);
      throw new Error(`Training WebSocket is not connected. Ready state: ${readyState}`);
    }
  };

  getStatus = () => {
    if (!this.ws) {
      return 'Not connected';
    }

    const readyStates = ['CONNECTING', 'OPEN', 'CLOSING', 'CLOSED'];
    return readyStates[this.ws.readyState] || 'UNKNOWN';
  };

  disconnect = () => {
    if (this.ws) {
      if (this.debugEnabled) console.log('[WebSocket Debug] Disconnecting WebSocket');
      this.shouldReconnect = false; // Prevent reconnection attempts when disconnecting intentionally
      this.ws.close(1000, "Client disconnecting"); // 1000 = normal closure
      this.ws = null;
      this.connectionState = 'disconnected';
      this.reconnectAttempts = 0;
    }
  };

  enableDebug = () => {
    this.debugEnabled = true;
    console.log('[WebSocket Debug] Debug mode enabled');
  };

  disableDebug = () => {
    this.debugEnabled = false;
    console.log('[WebSocket Debug] Debug mode disabled');
  };
}

// Store reference to test WebSocket manager
let testWsManager: TrainingWebSocketManager | null = null;

const wsManager = new TrainingWebSocketManager();

export const connectTrainingWebSocket = (onOutput: (data: any) => void) => {
  return wsManager.connect(onOutput);
};

export const sendTrainingConfig = (config: any) => {
  return wsManager.sendConfig(config);
};

export const connectTestWebSocket = (onOutput: (data: any) => void) => {
  // Create a new WebSocket manager for the test endpoint
  testWsManager = new TrainingWebSocketManager();
  return testWsManager.connect(onOutput, getWebSocketUrlForEndpoint('test'));
};

export const sendTestMessage = (message: string) => {
  if (testWsManager) {
    if (testWsManager.getStatus() === 'OPEN') {
      console.log('[WebSocket Debug] Sending test message:', message);
      testWsManager.sendRawMessage({ message });
    } else {
      throw new Error('Test WebSocket is not connected');
    }
  } else {
    throw new Error('Test WebSocket is not initialized');
  }
};

export const disconnectTestWebSocket = () => {
  if (testWsManager) {
    testWsManager.disconnect();
    testWsManager = null;
  }
};

export const getWebSocketStatus = () => {
  return wsManager.getStatus();
};

export const disconnectTrainingWebSocket = () => {
  return wsManager.disconnect();
};

export const enableTrainingWebSocketDebug = () => {
  wsManager.enableDebug();
};

export const disableTrainingWebSocketDebug = () => {
  wsManager.disableDebug();
};

// Specific API functions
export const runTraining = async (config: any) => {
  // Save the config first
  await saveConfig(config);

  // Then run training
  return apiCall('/api/run', {
    method: 'POST',
    body: JSON.stringify(config),
  });
};

export const saveConfig = async (config: any) => {
  return apiCall('/api/save_config', {
    method: 'POST',
    body: JSON.stringify(config),
  });
};

export const loadConfig = async (configPath: string) => {
  return apiCall(`/api/load_config?config_path=${encodeURIComponent(configPath)}`, {
    method: 'POST',
  });
};

export const loadTemplate = async (templateName: string, configPath: string) => {
  return apiCall('/api/load_template', {
    method: 'POST',
    body: JSON.stringify({ template_name: templateName, config_path: configPath }),
  });
};

export const listTemplates = async (configPath: string) => {
  return apiCall(`/api/list_templates?config_path=${encodeURIComponent(configPath)}`);
};

export const getAvailableScripts = async () => {
  return apiCall('/api/available_scripts');
};

export const getDefaultConfig = async () => {
  return apiCall('/api/default_config');
};

export const changeScript = async (script: string, configPath: string) => {
  return apiCall('/api/change_script', {
    method: 'POST',
    body: JSON.stringify({ script, config_path: configPath }),
  });
};

export const healthCheck = async () => {
  return apiCall('/api/health');
};

export const stopTraining = async () => {
  return apiCall('/api/stop_training', {
    method: 'POST',
  });
};