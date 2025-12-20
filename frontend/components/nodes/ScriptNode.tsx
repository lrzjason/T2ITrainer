import React, { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { BaseNode, outputHandleStyle } from './BaseNode';
import { SCRIPTS, TRANSLATIONS } from '../../constants';
import { useFlowContext } from '../FlowContext';

export const ScriptNode = memo(({ id, data, selected }: NodeProps) => {
  const { updateNodeData, lang } = useFlowContext();
  const t = TRANSLATIONS[lang];
  
  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    updateNodeData(id, 'scriptName', e.target.value);
  };

  const handleConfigPathChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    updateNodeData(id, 'configPath', e.target.value);
  }

  const handleRowClass = "relative flex items-center justify-between bg-white dark:bg-zinc-900/40 p-1.5 rounded mb-1 border border-zinc-200 dark:border-zinc-800 hover:border-zinc-400 dark:hover:border-zinc-700 transition-colors";
  const labelClass = "text-[10px] text-zinc-600 dark:text-zinc-400 font-medium pr-3";
  const inputClass = "w-full bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-zinc-700 rounded px-2 py-1 text-xs text-zinc-900 dark:text-zinc-200 focus:outline-none focus:border-indigo-500 dark:focus:border-indigo-800 transition-colors";
  const fieldLabelClass = "text-[10px] text-zinc-500 mb-1 block font-semibold";

  return (
    <BaseNode title={t.script} color="bg-indigo-950" selected={selected} id={id} data={data}>
      <div className="mb-4 space-y-3">
          <div>
            <label className={fieldLabelClass}>{t.python_script}</label>
            <select className={inputClass} value={data.scriptName} onChange={handleChange}>
              {SCRIPTS.map(s => (
                <option key={s.name} value={s.name}>{s.name} ({s.type})</option>
              ))}
            </select>
          </div>
          <div>
             <label className={fieldLabelClass}>{t.config_path}</label>
             <input type="text" className={`${inputClass} font-mono`}
                value={data.configPath || "config.json"} onChange={handleConfigPathChange} />
          </div>
      </div>

      <div className="border-t border-zinc-200 dark:border-zinc-800 pt-3 space-y-1">
        <div className={handleRowClass}>
          <Handle type="source" position={Position.Right} id="dir_out" style={outputHandleStyle} />
          <span className={labelClass}>{t.to_dir}</span>
        </div>
        <div className={handleRowClass}>
          <Handle type="source" position={Position.Right} id="lora_out" style={outputHandleStyle} />
          <span className={labelClass}>{t.to_lora}</span>
        </div>
        <div className={handleRowClass}>
          <Handle type="source" position={Position.Right} id="misc_out" style={outputHandleStyle} />
          <span className={labelClass}>{t.to_misc}</span>
        </div>
        <div className="h-px bg-zinc-200 dark:bg-zinc-800 my-2"></div>
        <div className={handleRowClass}>
          <Handle type="source" position={Position.Right} id="json_out" style={outputHandleStyle} />
          <span className={labelClass}>{t.to_ds_list}</span>
        </div>
        <div className={handleRowClass}>
          <Handle type="source" position={Position.Right} id="old_out" style={outputHandleStyle} />
          <span className={labelClass}>{t.to_old_schema}</span>
        </div>
      </div>
    </BaseNode>
  );
});