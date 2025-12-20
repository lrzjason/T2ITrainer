import React, { useState, useEffect, useRef } from 'react';
import { X, Send } from 'lucide-react';
import { connectTestWebSocket, sendTestMessage, disconnectTestWebSocket } from '../utils/api';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'backend';
  timestamp: Date;
}

export const ChatPanel: React.FC<{ onClose: () => void }> = ({ onClose }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isConnected, setIsConnected] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Connect to WebSocket when component mounts
  useEffect(() => {
    const handleMessage = (data: any) => {
      console.log('[ChatPanel] Received message:', data);
      
      if (data.type === 'output') {
        const newMessage: Message = {
          id: Date.now().toString(),
          text: data.data,
          sender: 'backend',
          timestamp: new Date()
        };
        setMessages(prev => [...prev, newMessage]);
      } else if (data.type === 'connection') {
        setIsConnected(data.status === 'connected');
        const statusMessage: Message = {
          id: Date.now().toString(),
          text: data.message,
          sender: 'backend',
          timestamp: new Date()
        };
        setMessages(prev => [...prev, statusMessage]);
      } else if (data.type === 'error') {
        const errorMessage: Message = {
          id: Date.now().toString(),
          text: `Error: ${data.message}`,
          sender: 'backend',
          timestamp: new Date()
        };
        setMessages(prev => [...prev, errorMessage]);
      }
    };

    // Connect to the test WebSocket endpoint
    connectTestWebSocket(handleMessage);
    
    // Add welcome message
    const welcomeMessage: Message = {
      id: 'welcome',
      text: 'Welcome to the chat! Type a message and press Enter to send it to the backend.',
      sender: 'backend',
      timestamp: new Date()
    };
    setMessages([welcomeMessage]);

    return () => {
      disconnectTestWebSocket();
    };
  }, []);

  const handleSend = () => {
    if (inputValue.trim() && isConnected) {
      // Add user message to chat
      const userMessage: Message = {
        id: Date.now().toString(),
        text: inputValue,
        sender: 'user',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, userMessage]);
      
      // Send to backend
      try {
        sendTestMessage(inputValue);
        setInputValue('');
      } catch (error) {
        const errorMessage: Message = {
          id: Date.now().toString(),
          text: `Failed to send message: ${error instanceof Error ? error.message : 'Unknown error'}`,
          sender: 'backend',
          timestamp: new Date()
        };
        setMessages(prev => [...prev, errorMessage]);
      }
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="absolute right-4 top-4 bottom-4 w-96 bg-white/30 dark:bg-zinc-900/40 backdrop-blur-sm border border-zinc-200 dark:border-zinc-800 rounded-2xl shadow-2xl flex flex-col z-40 animate-in slide-in-from-right-4 fade-in duration-200">
      <div className="p-4 border-b border-zinc-200 dark:border-zinc-800 flex justify-between items-center bg-white/70 dark:bg-zinc-900/70 rounded-t-2xl">
        <h2 className="text-zinc-600 dark:text-zinc-400 text-xs font-bold uppercase tracking-wider flex items-center gap-2">
          Chat Test
        </h2>
        <button onClick={onClose} className="text-zinc-400 hover:text-zinc-900 dark:hover:text-white">
          <X size={14} />
        </button>
      </div>
      
      <div className="flex-1 overflow-hidden flex flex-col">
        {/* Messages container */}
        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[80%] rounded-lg p-3 text-sm ${
                  message.sender === 'user'
                    ? 'bg-blue-500 text-white rounded-br-none'
                    : 'bg-zinc-200 dark:bg-zinc-800 text-zinc-800 dark:text-zinc-200 rounded-bl-none'
                }`}
              >
                <div className="whitespace-pre-wrap break-words">{message.text}</div>
                <div className={`text-xs mt-1 ${message.sender === 'user' ? 'text-blue-100' : 'text-zinc-500 dark:text-zinc-400'}`}>
                  {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
        
        {/* Input area */}
        <div className="p-4 border-t border-zinc-200 dark:border-zinc-800">
          <div className="flex gap-2">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={isConnected ? "Type a message..." : "Connecting..."}
              disabled={!isConnected}
              className="flex-1 bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-900 dark:text-zinc-100 placeholder-zinc-500 dark:placeholder-zinc-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50"
            />
            <button
              onClick={handleSend}
              disabled={!inputValue.trim() || !isConnected}
              className="bg-blue-500 hover:bg-blue-600 text-white rounded-lg p-2 flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Send size={16} />
            </button>
          </div>
          <div className="mt-2 text-xs text-zinc-500 dark:text-zinc-400 text-center">
            {isConnected ? 'Connected to backend' : 'Connecting to backend...'}
          </div>
        </div>
      </div>
    </div>
  );
};