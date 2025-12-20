import React from 'react';
import { Copy, Pin } from 'lucide-react';
import { NodeResizer } from 'reactflow';
import { useFlowContext } from '../FlowContext';

interface BaseNodeProps {
  title: string;
  color?: string;
  children: React.ReactNode;
  selected?: boolean;
  id?: string;
  data?: any; // data is now used for pinning state
}

export const BaseNode: React.FC<BaseNodeProps> = ({ 
  title, 
  color = "bg-zinc-700", 
  children,
  selected,
  id,
  data
}) => {
  const { copyNode, toggleLock } = useFlowContext();

  const containerClass = `shadow-xl rounded-lg border min-w-[250px] h-full flex flex-col relative
    bg-zinc-100 dark:bg-zinc-800 
    ${selected ? 'border-blue-500 ring-1 ring-blue-500/50' : 'border-zinc-300 dark:border-zinc-700'}`;

  const headerClass = `${color} px-2 py-1 flex items-center justify-between border-b border-black/10 dark:border-zinc-900/50 rounded-t-lg`;

  return (
    <div className={containerClass}>
      <NodeResizer 
        color="#3b82f6" 
        isVisible={!!selected} 
        minWidth={250} 
        minHeight={100}
      />

      {/* Header / Toolbar */}
      <div className={headerClass}>
        <span className="text-[10px] font-bold text-white tracking-wide">{title}</span>
        
        <div className="flex items-center gap-1">
            {id && (
                <>
                    <button onClick={() => toggleLock(id)} className={`p-1 hover:bg-black/20 rounded transition-colors ${data?.pinned ? 'text-blue-300' : 'text-zinc-100/70 hover:text-blue-200'}`} title={data?.pinned ? "Unpin Node" : "Pin Node"}>
                        <Pin size={10} fill={data?.pinned ? "currentColor" : "none"} />
                    </button>
                    <button onClick={() => copyNode(id)} className="p-1 hover:bg-black/20 rounded text-zinc-100/70 hover:text-blue-200 transition-colors" title="Clone Node">
                        <Copy size={10} />
                    </button>
                </>
            )}
        </div>
      </div>

      <div className="p-3 text-sm text-zinc-700 dark:text-zinc-200 space-y-2 flex-1 overflow-y-auto overflow-x-hidden scrollbar-thin min-h-0">
        {children}
      </div>
    </div>
  );
};

// Hit area is 24x24, visual dot is 12px (6px radius)
// We use radial-gradient to make the outer part transparent but clickable
export const inputHandleStyle = { 
    width: '24px', 
    height: '24px', 
    background: 'radial-gradient(circle, #60a5fa 6px, transparent 6.5px)', // blue
    border: 'none',
    left: '-12px',
    zIndex: 10
};

export const outputHandleStyle = { 
    width: '24px', 
    height: '24px', 
    background: 'radial-gradient(circle, #4ade80 6px, transparent 6.5px)', // green
    border: 'none',
    right: '-12px',
    zIndex: 10
};