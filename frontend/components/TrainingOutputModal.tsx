import React, { useState, useEffect, useRef, useCallback } from 'react';
import { X, Play, Square, Copy } from 'lucide-react';
import { connectTrainingWebSocket, sendTrainingConfig, disconnectTrainingWebSocket, stopTraining, enableTrainingWebSocketDebug, disableTrainingWebSocketDebug, getWebSocketStatus, connectTestWebSocket, sendTestMessage, disconnectTestWebSocket, getLog, getTrainingStatus, resetTrainingStatus } from '../utils/api';

interface TrainingOutputModalProps {
  isOpen: boolean;
  onClose: () => void;
  config: any;
  scriptName: string;
  configPath: string;
  onSaveWorkflow?: (name: string, workflow: { nodes: any[]; edges: any[] }) => Promise<void>;
  onTrainingStart?: () => void;
}

interface OutputMessage {
  type: string;
  data?: string;
  status?: string;
  return_code?: number;
  message?: string;
}

export const TrainingOutputModal: React.FC<TrainingOutputModalProps> = ({
  isOpen,
  onClose,
  config,
  scriptName,
  configPath
}) => {
  // Wrapper for onClose to ensure proper cleanup
  const handleClose = useCallback(() => {
    console.log('[Training Modal Debug] Closing modal, cleaning up connections');
    // Disconnect any active training connections
    disconnectTrainingWebSocket();
    disconnectTestWebSocket();
    // Reset state
    setIsRunning(false);
    setStatus('idle');
    setOutput([]);
    // Reset the backend training status when closing modal to ensure clean state
    resetTrainingStatus().catch((resetError) => {
      console.error('[Training Modal Debug] Error resetting training status on close:', resetError);
    });
    // Call the original onClose
    onClose();
  }, [onClose]);
  const [output, setOutput] = useState<string[]>([]);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [status, setStatus] = useState<'idle' | 'connecting' | 'running' | 'completed' | 'error'>('idle');
  const [wsReady, setWsReady] = useState<boolean>(false);
  const [debugMode, setDebugMode] = useState<boolean>(false); // Disable debug mode by default
  const [connectionStatus, setConnectionStatus] = useState<string>('Not connected');
  const [isTestMode, setIsTestMode] = useState<boolean>(false); // Track if we're in test mode
  const [isLoading, setIsLoading] = useState<boolean>(false); // Track loading state
  const outputRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [output]);

  // Separate effect for WebSocket connection management - only for cleanup
  useEffect(() => {
    // Enable debug mode if needed
    if (debugMode) {
      enableTrainingWebSocketDebug();
      console.log('[Training Modal Debug] Debug mode enabled');
    } else {
      disableTrainingWebSocketDebug();
    }

    // Clean up function to disconnect WebSocket when modal closes
    return () => {
      console.log('[Training Modal Debug] Cleaning up WebSocket connection');
      disconnectTrainingWebSocket();
      disconnectTestWebSocket(); // Also disconnect test WebSocket
      disableTrainingWebSocketDebug(); // Make sure to disable debug when closing
    };
  }, [isOpen, debugMode]); // Only re-run when modal opens/closes or debug mode changes

  // Effect to load previous logs when modal opens
  useEffect(() => {
    if (isOpen) {
      setIsLoading(true); // Set loading state when modal opens
      
      // Load previous logs and training status
      const loadPreviousLogs = async () => {
        try {
          // Get training status
          const statusResponse = await getTrainingStatus();
          console.log('[Training Modal Debug] Training status:', statusResponse);
          
          // If training is still running, we should connect to WebSocket
          if (statusResponse.training_status?.current_status === 'running') {
            setStatus('running');
            setIsRunning(true);
          } else if (statusResponse.training_status?.current_status === 'error' || 
                     statusResponse.training_status?.current_status === 'failed') {
            // If training has failed, set appropriate status but not running
            setStatus('error');
            setIsRunning(false);
            // Reset the backend training status so a new training can start
            try {
              await resetTrainingStatus();
            } catch (resetError) {
              console.error('[Training Modal Debug] Error resetting training status:', resetError);
            }
          } else if (statusResponse.training_status?.current_status === 'completed') {
            // If training has completed, set appropriate status but not running
            setStatus('completed');
            setIsRunning(false);
            // Reset the backend training status so a new training can start
            try {
              await resetTrainingStatus();
            } catch (resetError) {
              console.error('[Training Modal Debug] Error resetting training status:', resetError);
            }
          } else if (statusResponse.training_status?.current_status === 'idle' || statusResponse.training_status?.current_status === 'stopped') {
            // If training is idle or stopped, set appropriate status
            setStatus('idle');
            setIsRunning(false);
          }
          
          // Get today's logs
          const logResponse = await getLog();
          console.log('[Training Modal Debug] Log response:', logResponse);
          
          if (logResponse.logs && logResponse.logs.length > 0) {
            // Process logs and add to output
            const logLines = logResponse.logs
              .filter((entry: any) => entry.type === 'output')
              .map((entry: any) => entry.data);
            
            if (logLines.length > 0) {
              setOutput(logLines);
            }
          }
        } catch (error) {
          console.error('[Training Modal Debug] Error loading previous logs:', error);
        } finally {
          setIsLoading(false); // Reset loading state after data is loaded
        }
      };
      
      loadPreviousLogs();
    } else {
      // Reset loading state when modal closes
      setIsLoading(false);
    }
  }, [isOpen]);

  // Effect to handle debug mode changes
  useEffect(() => {
    if (debugMode) {
      enableTrainingWebSocketDebug();
      console.log('[Training Modal Debug] Debug mode enabled');
    } else {
      disableTrainingWebSocketDebug();
      console.log('[Training Modal Debug] Debug mode disabled');
    }
  }, [debugMode]);

  // Effect to monitor output changes
  useEffect(() => {
    if (debugMode) {
      console.log('[Training Modal Debug] Output updated, new length:', output.length);
    }
  }, [output, debugMode]);

  // Effect to monitor WebSocket status
  useEffect(() => {
    if (!isOpen) {
      return;
    }

    // Check connection status periodically
    const statusInterval = setInterval(() => {
      const currentStatus = getWebSocketStatus();
      setConnectionStatus(currentStatus);

      // Update wsReady based on actual WebSocket status
      const ready = currentStatus === 'OPEN';
      if (wsReady !== ready) {
        setWsReady(ready);
        console.log(`[Training Modal Debug] WebSocket ready state changed to: ${ready}`);
      }

      console.log(`[Training Modal Debug] WebSocket status: ${currentStatus}, Ready: ${ready}`);
    }, 500); // Check every 500ms

    return () => {
      clearInterval(statusInterval);
    };
  }, [isOpen, wsReady, debugMode]); // Added debugMode back to dependency array

  const sendConfigToTraining = (trainingConfig: any) => {
    try {
      console.log('[Training Modal Debug] Sending training config:', trainingConfig);
      sendTrainingConfig(trainingConfig);
      console.log('[Training Modal Debug] Training config sent successfully');
      // Don't add to output here, let the backend messages handle it
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      console.error('[Training Modal Debug] Error sending config:', error);
      setOutput(prev => [...prev, `Error sending config: ${errorMessage}`]);
      setStatus('error');
      setIsRunning(false);
      throw error; // Re-throw to be caught by caller
    }
  };

  const handleOutput = useCallback((data: OutputMessage) => {
    if (debugMode) {
      console.log('[TrainingOutputModal] Received data:', data);
    }
    
    // Handle raw data (in case of parsing errors)
    if (data.type === 'raw' && data.data) {
      if (debugMode) {
        console.log('[TrainingOutputModal] Processing raw message:', data.data);
      }
      setOutput(prev => [...prev, `[RAW] ${data.data}`]);
      setStatus('running');
      return;
    }
    
    if (data.type === 'output' && data.data) {
      if (debugMode) {
        console.log('[TrainingOutputModal] Processing output message:', data.data);
      }
      setOutput(prev => [...prev, data.data]);
      setStatus('running');
      // If we're in test mode, we can stop after receiving the response
      if (isTestMode) {
        setIsRunning(false);
      }
    } else if (data.type === 'complete') {
      if (debugMode) {
        console.log('[TrainingOutputModal] Processing complete message:', data);
      }
      setOutput(prev => [...prev, `Training ${data.status === 'success' ? 'completed' : 'failed'} with return code: ${data.return_code}`]);
      setStatus(data.status === 'success' ? 'completed' : 'error');
      setIsRunning(false);
    } else if (data.type === 'error') {
      if (debugMode) {
        console.log('[TrainingOutputModal] Processing error message:', data.message);
      }
      setOutput(prev => [...prev, `Error: ${data.message}`]);
      setStatus('error');
      setIsRunning(false);
    } else if (data.type === 'flush') {
      // Ignore flush messages, they're just for ensuring connection stays alive
      if (debugMode) {
        console.log('[TrainingOutputModal] Ignoring flush message');
      }
      return;
    } else if (data.type === 'connection') {
      // Handle connection acknowledgment
      if (debugMode) {
        console.log('[TrainingOutputModal] Processing connection message:', data.message);
      }
      setOutput(prev => [...prev, `Connected to training server: ${data.message}`]);
      setStatus('running');
    } else if (data.type === 'training_end') {
      // Handle training end message
      if (debugMode) {
        console.log('[TrainingOutputModal] Processing training end message:', data.data);
      }
      setOutput(prev => [...prev, data.data || 'Training completed']);
      setStatus('completed');
      setIsRunning(false);
    } else {      // Handle any other message types - still show the data even if type is unknown
      if (debugMode) {
        console.log('[TrainingOutputModal] Processing unknown message type:', data);
      }
      if (data.data) {
        setOutput(prev => [...prev, data.data]);
      } else if (typeof data === 'string') {
        // Handle case where data is a plain string
        setOutput(prev => [...prev, data]);
      } else {
        // Fallback for unknown data format
        setOutput(prev => [...prev, `Unknown message format: ${JSON.stringify(data)}`]);
      }
    }
    
    if (debugMode) {
      console.log('[TrainingOutputModal] Output state updated');
    }
  }, [debugMode, setOutput, setStatus, setIsRunning, isTestMode]);

  const startTraining = async () => {
    if (isTestMode) {
      console.log('[Training Modal Debug] Cannot start training while in test mode');
      return;
    }
    
    if (debugMode) {
      console.log('[Training Modal Debug] Start training called');
      console.log('[Training Modal Debug] Script name:', scriptName);
      console.log('[Training Modal Debug] Config path:', configPath);
    }

    if (!scriptName || !configPath) {
      setOutput(prev => [...prev, 'Error: Missing script name or config path']);
      setStatus('error');
      return;
    }

    try {
      // Call the training start callback if provided
      if (onTrainingStart) {
        onTrainingStart();
      }

      console.log('[Training Modal Debug] Starting training mode');
      setStatus('connecting');
      setOutput(['Starting training connection...']);
      setIsRunning(true);

      // Prepare the training configuration
      const trainingConfig = {
        script: scriptName,
        config_path: configPath,
        ...config
      };

      if (debugMode) {
        console.log('[Training Modal Debug] Prepared training config:', trainingConfig);
      }

      // Disconnect any existing connections first
      disconnectTrainingWebSocket();
      
      // Connect to the training WebSocket
      connectTrainingWebSocket(handleOutput);
      
      // Wait a bit for connection to establish, then send config
      setTimeout(() => {
        try {
          console.log('[Training Modal Debug] Sending training config');
          sendConfigToTraining(trainingConfig);
        } catch (error) {
          console.error('[Training Modal Debug] Error sending training config:', error);
          setOutput(prev => [...prev, `Error sending training config: ${error instanceof Error ? error.message : String(error)}`]);
          setStatus('error');
          setIsRunning(false);
        }
      }, 1000);
    } catch (error) {
      console.error('[Training Modal Debug] Error in startTraining:', error);
      setOutput(prev => [...prev, `Error starting training: ${error instanceof Error ? error.message : String(error)}`]);
      setStatus('error');
      setIsRunning(false);
    }
  };

  const stopCurrentTraining = async () => {
    try {
      await stopTraining();
      setOutput(prev => [...prev, 'Training stopped by user']);
      setIsRunning(false);
      setStatus('idle');
      // Disconnect the training WebSocket when stopping
      disconnectTrainingWebSocket();
    } catch (error) {
      setOutput(prev => [...prev, `Error stopping training: ${error instanceof Error ? error.message : String(error)}`]);
      setStatus('error');
    }
  };

  const clearOutput = () => {
    setOutput([]);
  };

  const copyOutput = () => {
    navigator.clipboard.writeText(output.join('\n'));
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
      <div className="bg-white dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 rounded-2xl shadow-2xl w-full max-w-4xl h-[80vh] flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-zinc-200 dark:border-zinc-800 bg-white/50 dark:bg-zinc-900/50 backdrop-blur-md">
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${
              status === 'running' ? 'bg-green-500 animate-pulse' :
              status === 'connecting' ? 'bg-yellow-500 animate-pulse' :
              status === 'completed' ? 'bg-blue-500' :
              status === 'error' ? 'bg-red-500' : 'bg-gray-400'
            }`}></div>
            <h2 className="text-lg font-bold text-zinc-900 dark:text-white">
              Training Output
            </h2>
            <span className="text-sm text-zinc-500 dark:text-zinc-400 capitalize">
              ({status})
            </span>
            <span className="ml-2 text-xs px-2 py-1 rounded bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-300">
              WS: {connectionStatus}
            </span>
            {debugMode && (
              <span className="ml-2 text-xs px-2 py-1 rounded bg-amber-100 dark:bg-amber-900 text-amber-800 dark:text-amber-200">
                DEBUG
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setDebugMode(!debugMode)}
              className={`w-12 h-8 flex items-center justify-center transition-colors border rounded-lg ${
                debugMode
                  ? 'bg-blue-100 dark:bg-blue-900/50 text-blue-600 dark:text-blue-300 border-blue-300 dark:border-blue-700'
                  : 'text-zinc-500 hover:text-zinc-800 dark:text-zinc-400 hover:dark:text-white border-zinc-200 dark:border-zinc-700'
              }`}
              title={debugMode ? "Disable debug mode" : "Enable debug mode"}
            >
              <span className="text-xs font-bold">Debug</span>
            </button>
            <button
              onClick={copyOutput}
              className="w-8 h-8 flex items-center justify-center text-zinc-500 hover:text-zinc-800 dark:text-zinc-400 hover:dark:text-white transition-colors border border-zinc-200 dark:border-zinc-700 rounded-lg"
              title="Copy output"
            >
              <Copy size={16} />
            </button>
            <button
              onClick={clearOutput}
              className="w-10 h-8 flex items-center justify-center text-zinc-500 hover:text-zinc-800 dark:text-zinc-400 hover:dark:text-white transition-colors border border-zinc-200 dark:border-zinc-700 rounded-lg text-xs font-medium"
              title="Clear output"
            >
              Clear
            </button>
            <button
              onClick={handleClose}
              className="w-8 h-8 flex items-center justify-center text-zinc-500 hover:text-red-500 dark:text-zinc-400 hover:dark:text-red-400 transition-colors border border-zinc-200 dark:border-zinc-700 rounded-lg"
              title="Close"
            >
              <X size={16} />
            </button>
          </div>
        </div>

        {/* Controls */}
        <div className="p-4 border-b border-zinc-200 dark:border-zinc-800 bg-white/30 dark:bg-zinc-900/30">
          <div className="flex items-center gap-3">
            {!isRunning ? (
              <button
                onClick={startTraining}
                disabled={status === 'connecting'}
                className="flex items-center gap-2 bg-emerald-500 hover:bg-emerald-600 text-white px-4 py-2 rounded-lg font-medium disabled:opacity-50 transition-colors"
              >
                <Play size={16} />
                {status === 'connecting' ? 'Connecting...' : 'Start Training'}
              </button>
            ) : (
              <button
                onClick={async () => {
                  if (isTestMode) {
                    // For test mode, just disconnect and reset
                    disconnectTestWebSocket();
                    setIsTestMode(false);
                    setIsRunning(false);
                    setStatus('idle');
                  } else if (status === 'completed' || status === 'error') {
                    // Training has already ended, reset the state and backend status
                    setIsRunning(false);
                    setStatus('idle');
                    // Reset the backend training status so a new training can start
                    try {
                      await resetTrainingStatus();
                    } catch (resetError) {
                      console.error('[Training Modal Debug] Error resetting training status:', resetError);
                    }
                  } else {
                    // Training is still running, stop it
                    stopCurrentTraining();
                  }
                }}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                  status === 'completed' || status === 'error'
                    ? 'bg-blue-500 hover:bg-blue-600 text-white'
                    : 'bg-red-500 hover:bg-red-600 text-white'
                }`}
              >
                <Square size={16} />
                {isTestMode ? 'Stop Test' : 
                 status === 'completed' ? 'Training Completed' : 
                 status === 'error' ? 'Training Failed' : 'Stop Training'}
              </button>
            )}
            <span className="text-sm text-zinc-600 dark:text-zinc-400">
              Script: {scriptName || 'Not selected'}
            </span>
          </div>
        </div>

        {/* Output Area */}
        <div className="flex-1 overflow-hidden bg-zinc-900 text-green-400 font-mono text-sm p-4">
          <div
            ref={outputRef}
            className="h-full overflow-y-auto font-mono whitespace-pre-wrap scrollbar-thin scrollbar-thumb-zinc-600 scrollbar-track-zinc-800"
          >
            {isLoading ? (
              <div className="h-full flex items-center justify-center text-zinc-500 dark:text-zinc-600">
                <div className="flex flex-col items-center">
                  <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500 mb-2"></div>
                  <span>Loading training data...</span>
                </div>
              </div>
            ) : output.length === 0 ? (
              <div className="h-full flex items-center justify-center text-zinc-500 dark:text-zinc-600">
                {status === 'idle'
                  ? 'Click "Start Training" to begin...'
                  : status === 'connecting'
                    ? 'Connecting to training server...'
                    : 'Waiting for output...'}
              </div>
            ) : (
              output.map((line, index) => (
                <div key={index} className="py-1">
                  {line}
                </div>
              ))
            )}
          </div>
        </div>

        {/* Status Bar */}
        <div className="p-3 border-t border-zinc-200 dark:border-zinc-800 bg-zinc-100 dark:bg-zinc-900/50 text-xs text-zinc-600 dark:text-zinc-400">
          {output.length} lines | WS: {connectionStatus} | {status === 'running' ? 'Live output' : 'Training finished'}
          {debugMode && (
            <div className="mt-1 text-[10px] text-zinc-500 dark:text-zinc-500">
              Debug: WebSocket ready = {wsReady ? 'true' : 'false'} | Is running = {isRunning ? 'true' : 'false'}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};