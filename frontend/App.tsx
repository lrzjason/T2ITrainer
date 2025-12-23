import React from 'react';
import { ReactFlowProvider } from 'reactflow';
import { FlowEditor } from './components/FlowEditor';
import { ToastProvider } from './components/ToastContext';

function App() {
  return (
    <ToastProvider>
      <ReactFlowProvider>
        <FlowEditor />
      </ReactFlowProvider>
    </ToastProvider>
  );
}

export default App;