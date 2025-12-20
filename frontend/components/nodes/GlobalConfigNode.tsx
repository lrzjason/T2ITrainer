import React, { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { BaseNode, outputHandleStyle } from './BaseNode';
import { OPTIMIZERS, PRECISIONS, SCHEDULERS } from '../../constants';

export const GlobalConfigNode = memo(({ data, selected }: NodeProps) => {

  const update = (key: string, val: any) => {
    data.onChange?.(key, val);
  };

  return (
    <BaseNode title="Global Hyperparameters" color="bg-blue-900" selected={selected}>
      
      <div className="max-h-[400px] overflow-y-auto pr-2 space-y-2 scrollbar-thin">
        
        <div>
            <label className="text-[10px] text-zinc-500 uppercase">Pretrained Model Path</label>
            <input 
                className="w-full bg-zinc-900 rounded border border-zinc-700 px-1 text-xs"
                value={data.pretrained_model_name_or_path || ''}
                onChange={(e) => update('pretrained_model_name_or_path', e.target.value)}
            />
        </div>

        <div className="grid grid-cols-2 gap-2">
            <div>
                <label className="text-[10px] text-zinc-500 uppercase">Save Name</label>
                <input 
                    className="w-full bg-zinc-900 rounded border border-zinc-700 px-1 text-xs"
                    value={data.save_name || ''}
                    onChange={(e) => update('save_name', e.target.value)}
                />
            </div>
            <div>
                <label className="text-[10px] text-zinc-500 uppercase">Seed</label>
                <input 
                    type="number"
                    className="w-full bg-zinc-900 rounded border border-zinc-700 px-1 text-xs"
                    value={data.seed}
                    onChange={(e) => update('seed', parseInt(e.target.value))}
                />
            </div>
        </div>

        <div>
            <label className="text-[10px] text-zinc-500 uppercase">Output Dir</label>
            <input 
                className="w-full bg-zinc-900 rounded border border-zinc-700 px-1 text-xs"
                value={data.output_dir}
                onChange={(e) => update('output_dir', e.target.value)}
            />
        </div>

        <div className="grid grid-cols-2 gap-2">
            <div>
                <label className="text-[10px] text-zinc-500 uppercase">Rank</label>
                <input 
                    type="number"
                    className="w-full bg-zinc-900 rounded border border-zinc-700 px-1 text-xs"
                    value={data.rank}
                    onChange={(e) => update('rank', parseInt(e.target.value))}
                />
            </div>
             <div>
                <label className="text-[10px] text-zinc-500 uppercase">Resolution</label>
                <input 
                    className="w-full bg-zinc-900 rounded border border-zinc-700 px-1 text-xs"
                    value={data.resolution}
                    onChange={(e) => update('resolution', e.target.value)}
                />
            </div>
        </div>

        <div className="grid grid-cols-2 gap-2">
            <div>
                <label className="text-[10px] text-zinc-500 uppercase">Epochs</label>
                <input 
                    type="number"
                    className="w-full bg-zinc-900 rounded border border-zinc-700 px-1 text-xs"
                    value={data.num_train_epochs}
                    onChange={(e) => update('num_train_epochs', parseInt(e.target.value))}
                />
            </div>
            <div>
                <label className="text-[10px] text-zinc-500 uppercase">Batch Size</label>
                <input 
                    type="number"
                    className="w-full bg-zinc-900 rounded border border-zinc-700 px-1 text-xs"
                    value={data.train_batch_size}
                    onChange={(e) => update('train_batch_size', parseInt(e.target.value))}
                />
            </div>
        </div>
        
        <div className="grid grid-cols-2 gap-2">
            <div>
                <label className="text-[10px] text-zinc-500 uppercase">Optimizer</label>
                <select 
                    className="w-full bg-zinc-900 rounded border border-zinc-700 px-1 text-xs"
                    value={data.optimizer}
                    onChange={(e) => update('optimizer', e.target.value)}
                >
                    {OPTIMIZERS.map(o => <option key={o} value={o}>{o}</option>)}
                </select>
            </div>
            <div>
                <label className="text-[10px] text-zinc-500 uppercase">Precision</label>
                <select 
                    className="w-full bg-zinc-900 rounded border border-zinc-700 px-1 text-xs"
                    value={data.mixed_precision}
                    onChange={(e) => update('mixed_precision', e.target.value)}
                >
                    {PRECISIONS.map(o => <option key={o} value={o}>{o}</option>)}
                </select>
            </div>
        </div>

        <div>
            <label className="text-[10px] text-zinc-500 uppercase">Learning Rate</label>
            <input 
                type="number"
                className="w-full bg-zinc-900 rounded border border-zinc-700 px-1 text-xs"
                value={data.learning_rate}
                onChange={(e) => update('learning_rate', parseFloat(e.target.value))}
            />
        </div>

         <div className="flex items-center gap-2">
            <input 
                type="checkbox"
                checked={data.gradient_checkpointing || false}
                onChange={(e) => update('gradient_checkpointing', e.target.checked)}
            />
            <label className="text-[10px] text-zinc-500 uppercase">Gradient Checkpointing</label>
        </div>
      </div>

      <div className="flex justify-end mt-4 pt-2 border-t border-zinc-700">
        <span className="text-[10px] mr-3 text-zinc-400">Connect to Script</span>
        <Handle type="source" position={Position.Right} id="out" style={outputHandleStyle} />
      </div>

    </BaseNode>
  );
});