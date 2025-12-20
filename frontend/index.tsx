import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

// Suppress ResizeObserver loop errors which are often benign in React Flow / Resizable contexts
const isResizeObserverError = (msg: string) => {
    return msg.includes('ResizeObserver loop') || msg.includes('ResizeObserver loop completed with undelivered notifications.');
};

window.addEventListener('error', (e) => {
  const msg = e.message || '';
  if (typeof msg === 'string' && isResizeObserverError(msg)) {
    e.stopImmediatePropagation();
    e.preventDefault();
  }
});

const originalError = console.error;
console.error = (...args) => {
    if (args.length > 0) {
        // Check string argument
        if (typeof args[0] === 'string' && isResizeObserverError(args[0])) return;
        // Check Error object argument
        if (args[0] instanceof Error && isResizeObserverError(args[0].message)) return;
    }
    originalError.call(console, ...args);
};

const rootElement = document.getElementById('root');
if (!rootElement) {
  throw new Error("Could not find root element to mount to");
}

const root = ReactDOM.createRoot(rootElement);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);