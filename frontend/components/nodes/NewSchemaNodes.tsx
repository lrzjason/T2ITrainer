import React, { memo, useState, useEffect } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { BaseNode, outputHandleStyle, inputHandleStyle } from './BaseNode';
import { Plus, Trash2 } from 'lucide-react';
import { useFlowContext } from '../FlowContext';
import { TRANSLATIONS } from '../../constants';

const inputClass = "w-full bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-zinc-700 rounded px-2 py-1 text-xs text-zinc-900 dark:text-zinc-200 focus:outline-none focus:border-cyan-600 transition-colors";
const labelClass = "text-[9px] text-zinc-500 font-semibold mb-0.5 block";
const checkboxClass = "appearance-none w-3.5 h-3.5 border border-zinc-400 dark:border-zinc-600 rounded bg-white dark:bg-zinc-900 checked:bg-cyan-600 checked:border-cyan-600 relative cursor-pointer after:content-[''] after:absolute after:hidden after:checked:block after:left-[3px] after:top-[0px] after:w-[5px] after:h-[9px] after:border-r-2 after:border-b-2 after:border-white after:rotate-45";

// --- Dataset List Node (Aggregator) ---
export const DatasetListNode = memo(({ id, data, selected }: NodeProps) => {
    const { updateNodeData, lang, removeEdgesBySourceHandle } = useFlowContext();
    const t = TRANSLATIONS[lang];
    const [slots, setSlots] = useState<string[]>(data.slots || ['dataset_0']);

    useEffect(() => {
        if (!data.slots) {
            updateNodeData(id, 'slots', slots);
        }
    }, []);

    const updateSlots = (newSlots: string[]) => {
        setSlots(newSlots);
        updateNodeData(id, 'slots', newSlots);
    };

    const addSlot = () => {
        const newSlot = `dataset_${Math.random().toString(36).substr(2, 5)}`;
        updateSlots([...slots, newSlot]);
    };

    const removeSlot = (slotToRemove: string) => {
        // Remove the slot from the local state
        const filteredSlots = slots.filter(s => s !== slotToRemove);
        updateSlots(filteredSlots);
        
        // Remove any edges connected to this slot
        removeEdgesBySourceHandle(id, slotToRemove);
    };

    return (
        <BaseNode title={t.ds_list} color="bg-cyan-950" selected={selected} id={id} data={data}>
             <div className="flex justify-start mb-2 pb-2 border-b border-zinc-200 dark:border-zinc-700">
                 <Handle type="target" position={Position.Left} id="in" style={inputHandleStyle} />
                 <span className="text-[10px] ml-3 text-zinc-500 dark:text-zinc-400">{t.from_script}</span>
             </div>
             <div className="space-y-1">
                {slots.map((slotId, index) => (
                    <div key={slotId} className="relative flex items-center justify-between bg-white dark:bg-zinc-900/50 p-1 rounded border border-zinc-200 dark:border-zinc-800 hover:border-zinc-400 dark:hover:border-zinc-600 transition-colors">
                         <div className="flex items-center h-5 w-full justify-end">
                            <span className="text-[10px] text-zinc-700 dark:text-zinc-300 mr-3">{t.dataset} #{index + 1}</span>
                            <Handle type="source" position={Position.Right} id={slotId} style={outputHandleStyle} />
                        </div>
                        {slots.length > 1 && (
                            <button onClick={() => removeSlot(slotId)} className="absolute left-1 text-zinc-400 hover:text-red-500 dark:text-zinc-600 dark:hover:text-red-400">
                                <Trash2 size={10} />
                            </button>
                        )}
                    </div>
                ))}
            </div>
            <button onClick={addSlot} className="w-full bg-zinc-100 dark:bg-zinc-800 hover:bg-zinc-200 dark:hover:bg-zinc-700 rounded py-1 mt-2 text-[10px] flex justify-center items-center gap-1 border border-zinc-300 dark:border-zinc-700 text-zinc-700 dark:text-zinc-300"><Plus size={10}/> {t.add_slot}</button>
        </BaseNode>
    );
});

// --- Dataset Config Node ---
export const DatasetConfigNode = memo(({ id, data, selected }: NodeProps) => {
    const { updateNodeData, lang } = useFlowContext();
    const t = TRANSLATIONS[lang];
    const update = (k: string, v: any) => updateNodeData(id, k, v);

    const handleRowClass = "relative flex items-center justify-between bg-white dark:bg-zinc-900/40 p-1.5 rounded mb-1 border border-zinc-200 dark:border-zinc-800 hover:border-zinc-400 dark:hover:border-zinc-700 transition-colors";
    const labelRowClass = "text-[10px] text-zinc-600 dark:text-zinc-400 font-medium pr-3";

    return (
        <BaseNode title={t.ds_conf} color="bg-cyan-900" selected={selected} id={id} data={data}>
            <div className="flex justify-start mb-2 pb-2 border-b border-zinc-200 dark:border-zinc-700">
                 <Handle type="target" position={Position.Left} id="in" style={inputHandleStyle} />
                 <span className="text-[10px] ml-3 text-zinc-500 dark:text-zinc-400">{t.from_ds_list}</span>
             </div>
            
            <div className="space-y-2 mb-3">
                <div>
                    <label className={labelClass}>{t.data_dir}</label>
                    <input className={inputClass} value={data.train_data_dir || ""} onChange={e => update('train_data_dir', e.target.value)} />
                </div>
                <div className="grid grid-cols-2 gap-2">
                     <div>
                        <label className={labelClass}>{t.repeats}</label>
                        <input type="number" className={inputClass} value={data.repeats ?? 1} onChange={e => update('repeats', parseInt(e.target.value))} />
                    </div>
                     <div>
                        <label className={labelClass}>{t.resolution}</label>
                        <input className={inputClass} value={data.resolution || ""} placeholder="global" onChange={e => update('resolution', e.target.value)} />
                    </div>
                </div>

                <div className="space-y-1 pt-1">
                    <div className="text-[9px] text-zinc-500 font-semibold mb-1">{lang === 'en' ? 'Recreate' : '重新创建'}</div>
                     <div className="flex items-center gap-2">
                        <input type="checkbox" className={checkboxClass} checked={data.recreate_cache_target || false} onChange={e => update('recreate_cache_target', e.target.checked)} />
                        <label className="text-[9px] text-zinc-700 dark:text-zinc-300">{t.recreate_cache_target}</label>
                    </div>
                     <div className="flex items-center gap-2">
                        <input type="checkbox" className={checkboxClass} checked={data.recreate_cache_reference || false} onChange={e => update('recreate_cache_reference', e.target.checked)} />
                        <label className="text-[9px] text-zinc-700 dark:text-zinc-300">{t.recreate_cache_ref}</label>
                    </div>
                     <div className="flex items-center gap-2">
                        <input type="checkbox" className={checkboxClass} checked={data.recreate_cache_caption || false} onChange={e => update('recreate_cache_caption', e.target.checked)} />
                        <label className="text-[9px] text-zinc-700 dark:text-zinc-300">{t.recreate_cache_cap}</label>
                    </div>
                </div>
            </div>

            <div className="border-t border-zinc-200 dark:border-zinc-800 pt-2">
                 <div className="text-[9px] text-zinc-500 mb-2">{t.connect_multiple}</div>
                 <div className={handleRowClass}>
                    <Handle type="source" position={Position.Right} id="image_config" style={outputHandleStyle} />
                    <span className={labelRowClass}>{t.image_config_list}</span>
                 </div>
                 <div className={handleRowClass}>
                    <Handle type="source" position={Position.Right} id="target_config" style={outputHandleStyle} />
                    <span className={labelRowClass}>{t.target_config_list}</span>
                 </div>
                 <div className={handleRowClass}>
                    <Handle type="source" position={Position.Right} id="reference_config" style={outputHandleStyle} />
                    <span className={labelRowClass}>{t.reference_config_list}</span>
                 </div>
                 <div className={handleRowClass}>
                    <Handle type="source" position={Position.Right} id="caption_config" style={outputHandleStyle} />
                    <span className={labelRowClass}>{t.caption_config_list}</span>
                 </div>
                 <div className={handleRowClass}>
                    <Handle type="source" position={Position.Right} id="batch_config" style={outputHandleStyle} />
                    <span className={labelRowClass}>{t.batch_config_list}</span>
                 </div>
            </div>
        </BaseNode>
    );
});