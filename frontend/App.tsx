import React from 'react';
import { ReactFlowProvider } from 'reactflow';
import { FlowEditor } from './components/FlowEditor';

function App() {
  return (
    <ReactFlowProvider>
      <FlowEditor />
    </ReactFlowProvider>
  );
}

export default App;