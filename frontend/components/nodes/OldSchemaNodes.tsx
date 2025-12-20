import React, { memo, useState, useEffect } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { BaseNode, outputHandleStyle, inputHandleStyle } from './BaseNode';
import { useFlowContext } from '../FlowContext';
import { TRANSLATIONS } from '../../constants';
import { Plus, Trash2 } from 'lucide-react';

const inputClass = "w-full bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-zinc-700 rounded px-2 py-1 text-xs text-zinc-900 dark:text-zinc-200 focus:outline-none focus:border-amber-600 transition-colors";
const labelClass = "text-[9px] text-zinc-500 font-semibold mb-0.5 block";
const checkboxClass = "appearance-none w-3.5 h-3.5 border border-zinc-400 dark:border-zinc-600 rounded bg-white dark:bg-zinc-900 checked:bg-amber-600 checked:border-amber-600 relative cursor-pointer after:content-[''] after:absolute after:hidden after:checked:block after:left-[3px] after:top-[0px] after:w-[5px] after:h-[9px] after:border-r-2 after:border-b-2 after:border-white after:rotate-45";

// --- Helper for Dynamic Lists ---
const OldSchemaList = ({ id, data, selected, title, prefix }: any) => {
    const { updateNodeData, lang } = useFlowContext();
    const t = TRANSLATIONS[lang];
    const [slots, setSlots] = useState<string[]>(data.slots || [`${prefix}_0`]);

    useEffect(() => { if (!data.slots) updateNodeData(id, 'slots', slots); }, []);

    const updateSlots = (newSlots: string[]) => {
        setSlots(newSlots);
        updateNodeData(id, 'slots', newSlots);
    };

    const addSlot = () => updateSlots([...slots, `${prefix}_${Math.random().toString(36).substr(2, 5)}`]);
    const removeSlot = (slot: string) => updateSlots(slots.filter(s => s !== slot));

    return (
        <BaseNode title={title} color="bg-amber-900" selected={selected} id={id} data={data}>
             <div className="flex justify-start mb-2 pb-2 border-b border-zinc-200 dark:border-zinc-700">
                 <Handle type="target" position={Position.Left} id="in" style={inputHandleStyle} />
                 <span className="text-[10px] ml-3 text-zinc-500 dark:text-zinc-400">{t.from_root}</span>
             </div>
             <div className="space-y-1">
                {slots.map((slotId, index) => (
                    <div key={slotId} className="relative flex items-center justify-between bg-white dark:bg-zinc-900/50 p-1 rounded border border-zinc-200 dark:border-zinc-800 hover:border-zinc-400 dark:hover:border-zinc-600 transition-colors">
                         <div className="flex items-center h-5 w-full justify-end">
                            <span className="text-[10px] text-zinc-700 dark:text-zinc-300 mr-3">{t.item} #{index + 1}</span>
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
};

// --- ROOT NODE ---
export const OldSchemaRootNode = memo(({ id, data, selected }: NodeProps) => {
    const { lang } = useFlowContext();
    const t = TRANSLATIONS[lang];
    const rowClass = "relative flex justify-between items-center bg-white dark:bg-zinc-900/50 p-2 rounded border border-zinc-200 dark:border-zinc-800 hover:border-zinc-400 dark:hover:border-zinc-700 transition-colors";
    
    return (
        <BaseNode title={t.old_root} color="bg-amber-950" selected={selected} id={id} data={data}>
             <div className="flex justify-start mb-3 pb-2 border-b border-zinc-200 dark:border-zinc-700">
                 <Handle type="target" position={Position.Left} id="in" style={inputHandleStyle} />
                 <span className="text-[10px] ml-3 text-zinc-500 dark:text-zinc-400">{t.from_script}</span>
             </div>
             <div className="space-y-3">
                 <div className={rowClass}>
                     <Handle type="source" position={Position.Right} id="image_configs" style={outputHandleStyle} />
                     <span className="text-xs pr-3 text-zinc-700 dark:text-zinc-300">{t.old_image_list}</span>
                 </div>
                 <div className={rowClass}>
                     <Handle type="source" position={Position.Right} id="caption_configs" style={outputHandleStyle} />
                     <span className="text-xs pr-3 text-zinc-700 dark:text-zinc-300">{t.old_caption_list}</span>
                 </div>
                 <div className={rowClass}>
                     <Handle type="source" position={Position.Right} id="training_set" style={outputHandleStyle} />
                     <span className="text-xs pr-3 text-zinc-700 dark:text-zinc-300">{t.training_set_list}</span>
                 </div>
             </div>
        </BaseNode>
    );
});

// --- IMAGE NODES ---
export const OldImageListNode = memo((props: NodeProps) => {
    const { lang } = useFlowContext();
    return <OldSchemaList {...props} title={TRANSLATIONS[lang].old_image_list} prefix="old_img" />;
});

export const OldImageItemNode = memo(({ id, data, selected }: NodeProps) => {
    const { updateNodeData, lang } = useFlowContext();
    const t = TRANSLATIONS[lang];
    const update = (k: string, v: any) => updateNodeData(id, k, v);
    return (
        <BaseNode title={t.old_image_item} color="bg-amber-800" selected={selected} id={id} data={data}>
            <div className="flex justify-start mb-2 pb-2 border-b border-zinc-200 dark:border-zinc-700">
                 <Handle type="target" position={Position.Left} id="in" style={inputHandleStyle} />
                 <span className="text-[10px] ml-3 text-zinc-500 dark:text-zinc-400">{t.from_list}</span>
            </div>
            <div className="grid grid-cols-2 gap-2">
                <div>
                    <label className={labelClass}>{t.key}</label>
                    <input className={inputClass} value={data.key || ""} onChange={e => update('key', e.target.value)} placeholder="train" />
                </div>
                <div>
                    <label className={labelClass}>{t.suffix}</label>
                    <input className={inputClass} value={data.suffix || ""} onChange={e => update('suffix', e.target.value)} placeholder="_H" />
                </div>
            </div>
        </BaseNode>
    );
});

// --- CAPTION NODES ---
export const OldCaptionListNode = memo((props: NodeProps) => {
    const { lang } = useFlowContext();
    return <OldSchemaList {...props} title={TRANSLATIONS[lang].old_caption_list} prefix="old_cap" />;
});

export const OldCaptionItemNode = memo(({ id, data, selected }: NodeProps) => {
    const { updateNodeData, lang } = useFlowContext();
    const t = TRANSLATIONS[lang];
    const update = (k: string, v: any) => updateNodeData(id, k, v);
    return (
        <BaseNode title={t.old_caption_item} color="bg-amber-800" selected={selected} id={id} data={data}>
            <div className="flex justify-start mb-2 pb-2 border-b border-zinc-200 dark:border-zinc-700">
                 <Handle type="target" position={Position.Left} id="in" style={inputHandleStyle} />
                 <span className="text-[10px] ml-3 text-zinc-500 dark:text-zinc-400">{t.from_list}</span>
            </div>
            <div className="space-y-2">
                <div className="grid grid-cols-2 gap-2">
                    <div>
                        <label className={labelClass}>{t.key}</label>
                        <input className={inputClass} value={data.key || ""} onChange={e => update('key', e.target.value)} placeholder="train" />
                    </div>
                    <div>
                        <label className={labelClass}>{t.ext}</label>
                        <input className={inputClass} value={data.ext || ""} onChange={e => update('ext', e.target.value)} placeholder=".txt" />
                    </div>
                </div>
                <div>
                    <label className={labelClass}>{t.instruction}</label>
                    <textarea className={`${inputClass} h-16`} value={data.instruction || ""} onChange={e => update('instruction', e.target.value)} />
                </div>
                
                <div className="flex items-center justify-between bg-zinc-50 dark:bg-zinc-900/30 p-2 rounded border border-zinc-200 dark:border-zinc-800 mt-2">
                     <span className="text-[10px] text-zinc-500">{t.old_ref_list}</span>
                     <Handle type="source" position={Position.Right} id="ref_list" style={outputHandleStyle} />
                </div>
            </div>
        </BaseNode>
    );
});

// --- REF NODES (Nested in Caption) ---
export const OldRefImageListNode = memo((props: NodeProps) => {
    const { lang } = useFlowContext();
    // Reusing the generic list style, slightly different title logic
    return (
        <BaseNode title={TRANSLATIONS[lang].old_ref_list} color="bg-orange-900" selected={props.selected} id={props.id} data={props.data}>
             <div className="flex justify-start mb-2 pb-2 border-b border-zinc-200 dark:border-zinc-700">
                 <Handle type="target" position={Position.Left} id="in" style={inputHandleStyle} />
                 <span className="text-[10px] ml-3 text-zinc-500 dark:text-zinc-400">{TRANSLATIONS[lang].from_item}</span>
             </div>
             {/* Re-implement logic since we can't reuse OldSchemaList directly without modifying its handle text */}
             <OldSchemaListContent {...props} prefix="old_ref" />
        </BaseNode>
    );
});

// Internal content for manual lists to avoid HOC complexity
const OldSchemaListContent = ({ id, data, prefix }: any) => {
    const { updateNodeData, lang } = useFlowContext();
    const t = TRANSLATIONS[lang];
    const [slots, setSlots] = useState<string[]>(data.slots || [`${prefix}_0`]);
    useEffect(() => { if (!data.slots) updateNodeData(id, 'slots', slots); }, []);
    const updateSlots = (newSlots: string[]) => { setSlots(newSlots); updateNodeData(id, 'slots', newSlots); };
    const addSlot = () => updateSlots([...slots, `${prefix}_${Math.random().toString(36).substr(2, 5)}`]);
    const removeSlot = (slot: string) => updateSlots(slots.filter(s => s !== slot));

    return (
        <>
            <div className="space-y-1">
                {slots.map((slotId, index) => (
                    <div key={slotId} className="relative flex items-center justify-between bg-white dark:bg-zinc-900/50 p-1 rounded border border-zinc-200 dark:border-zinc-800 hover:border-zinc-400 dark:hover:border-zinc-600 transition-colors">
                         <div className="flex items-center h-5 w-full justify-end">
                            <span className="text-[10px] text-zinc-700 dark:text-zinc-300 mr-3">{t.item} #{index + 1}</span>
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
        </>
    )
}

export const OldRefImageItemNode = memo(({ id, data, selected }: NodeProps) => {
    const { updateNodeData, lang } = useFlowContext();
    const t = TRANSLATIONS[lang];
    const update = (k: string, v: any) => updateNodeData(id, k, v);
    return (
        <BaseNode title={t.old_ref_item} color="bg-orange-800" selected={selected} id={id} data={data}>
            <div className="flex justify-start mb-2 pb-2 border-b border-zinc-200 dark:border-zinc-700">
                 <Handle type="target" position={Position.Left} id="in" style={inputHandleStyle} />
                 <span className="text-[10px] ml-3 text-zinc-500 dark:text-zinc-400">{t.from_list}</span>
            </div>
            <div className="space-y-2">
                <div>
                    <label className={labelClass}>{t.target}</label>
                    <input className={inputClass} value={data.target || ""} onChange={e => update('target', e.target.value)} placeholder="10L" />
                </div>
                <div className="grid grid-cols-2 gap-2">
                     <div>
                        <label className={labelClass}>{t.dropout}</label>
                        <input type="number" step="0.1" className={inputClass} value={data.dropout} onChange={e => update('dropout', parseFloat(e.target.value))} placeholder="0.4" />
                    </div>
                    <div className="flex items-end pb-1">
                        <label className="flex items-center gap-2 cursor-pointer text-[10px] text-zinc-700 dark:text-zinc-300">
                            <input type="checkbox" className={checkboxClass} checked={data.resize || false} onChange={e => update('resize', e.target.checked)} />
                            {t.resize}
                        </label>
                    </div>
                </div>
            </div>
        </BaseNode>
    );
});

// --- TRAINING SET NODES ---
export const OldTrainingSetListNode = memo((props: NodeProps) => {
    const { lang } = useFlowContext();
    return <OldSchemaList {...props} title={TRANSLATIONS[lang].old_train_list} prefix="old_ts" />;
});

export const OldTrainingSetItemNode = memo(({ id, data, selected }: NodeProps) => {
    const { updateNodeData, lang } = useFlowContext();
    const t = TRANSLATIONS[lang];
    const update = (k: string, v: any) => updateNodeData(id, k, v);
    return (
        <BaseNode title={t.old_train_item} color="bg-indigo-800" selected={selected} id={id} data={data}>
            <div className="flex justify-start mb-2 pb-2 border-b border-zinc-200 dark:border-zinc-700">
                 <Handle type="target" position={Position.Left} id="in" style={inputHandleStyle} />
                 <span className="text-[10px] ml-3 text-zinc-500 dark:text-zinc-400">{t.from_list}</span>
            </div>
            <div className="space-y-2">
                <div>
                    <label className={labelClass}>{t.captions_selection_target}</label>
                    <input className={inputClass} value={data.captions_selection_target || ""} onChange={e => update('captions_selection_target', e.target.value)} placeholder="train" />
                </div>
                 <div className="flex items-center justify-between bg-zinc-50 dark:bg-zinc-900/30 p-2 rounded border border-zinc-200 dark:border-zinc-800 mt-2">
                     <span className="text-[10px] text-zinc-500">{t.old_layout_list}</span>
                     <Handle type="source" position={Position.Right} id="layout_list" style={outputHandleStyle} />
                </div>
            </div>
        </BaseNode>
    );
});

// --- LAYOUT NODES ---
export const OldLayoutListNode = memo((props: NodeProps) => {
     const { lang } = useFlowContext();
     return (
        <BaseNode title={TRANSLATIONS[lang].old_layout_list} color="bg-teal-900" selected={props.selected} id={props.id} data={props.data}>
             <div className="flex justify-start mb-2 pb-2 border-b border-zinc-200 dark:border-zinc-700">
                 <Handle type="target" position={Position.Left} id="in" style={inputHandleStyle} />
                 <span className="text-[10px] ml-3 text-zinc-500 dark:text-zinc-400">{TRANSLATIONS[lang].from_item}</span>
             </div>
             <OldSchemaListContent {...props} prefix="old_lay" />
        </BaseNode>
    );
});

export const OldLayoutItemNode = memo(({ id, data, selected }: NodeProps) => {
    const { updateNodeData, lang } = useFlowContext();
    const t = TRANSLATIONS[lang];
    const update = (k: string, v: any) => updateNodeData(id, k, v);
    return (
        <BaseNode title={t.old_layout_item} color="bg-teal-800" selected={selected} id={id} data={data}>
            <div className="flex justify-start mb-2 pb-2 border-b border-zinc-200 dark:border-zinc-700">
                 <Handle type="target" position={Position.Left} id="in" style={inputHandleStyle} />
                 <span className="text-[10px] ml-3 text-zinc-500 dark:text-zinc-400">{t.from_list}</span>
            </div>
             <div className="space-y-2">
                 <div className="grid grid-cols-2 gap-2">
                     <div>
                        <label className={labelClass}>{t.key}</label>
                        <input className={inputClass} value={data.key || ""} onChange={e => update('key', e.target.value)} placeholder="train" />
                    </div>
                     <div>
                        <label className={labelClass}>{t.target}</label>
                        <input className={inputClass} value={data.target || ""} onChange={e => update('target', e.target.value)} placeholder="train" />
                    </div>
                 </div>
                 <div className="grid grid-cols-2 gap-2">
                     <div>
                        <label className={labelClass}>{t.dropout}</label>
                        <input type="number" step="0.1" className={inputClass} value={data.dropout} onChange={e => update('dropout', parseFloat(e.target.value))} />
                    </div>
                     <div className="flex items-end pb-1">
                        <label className="flex items-center gap-2 cursor-pointer text-[10px] text-zinc-700 dark:text-zinc-300">
                            <input type="checkbox" className={checkboxClass} checked={data.noised || false} onChange={e => update('noised', e.target.checked)} />
                            {t.noised}
                        </label>
                    </div>
                 </div>
            </div>
        </BaseNode>
    );
});