import React, { memo, useState, useEffect } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { BaseNode, outputHandleStyle, inputHandleStyle } from './BaseNode';
import { Plus, Trash2 } from 'lucide-react';
import { useFlowContext } from '../FlowContext';
import { TRANSLATIONS } from '../../constants';

const inputClass = "bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-zinc-700 rounded px-2 py-1 text-xs text-zinc-900 dark:text-zinc-200 focus:outline-none focus:border-zinc-500 transition-colors w-full";
const labelClass = "text-[9px] text-zinc-500 font-semibold mb-0.5 block";
const checkboxClass = "appearance-none w-3.5 h-3.5 border border-zinc-400 dark:border-zinc-600 rounded bg-white dark:bg-zinc-900 checked:bg-pink-600 checked:border-pink-600 relative cursor-pointer after:content-[''] after:absolute after:hidden after:checked:block after:left-[3px] after:top-[0px] after:w-[5px] after:h-[9px] after:border-r-2 after:border-b-2 after:border-white after:rotate-45";

// --- Generic List Node Component ---
const GenericListNode = ({ data, selected, id, titleKey, color, prefix }: any) => {
    const { updateNodeData, lang } = useFlowContext();
    const t = TRANSLATIONS[lang];
    const [slots, setSlots] = useState<string[]>(data.slots || [`${prefix}_0`]);

    useEffect(() => {
        if (!data.slots) updateNodeData(id, 'slots', slots);
    }, []);

    const updateSlots = (newSlots: string[]) => {
        setSlots(newSlots);
        updateNodeData(id, 'slots', newSlots);
    };

    const addSlot = () => updateSlots([...slots, `${prefix}_${Math.random().toString(36).substr(2, 5)}`]);
    const removeSlot = (slot: string) => updateSlots(slots.filter(s => s !== slot));

    return (
        <BaseNode title={t[titleKey]} color={color} selected={selected} id={id}>
             <div className="flex justify-start mb-2 pb-2 border-b border-zinc-200 dark:border-zinc-700">
                 <Handle type="target" position={Position.Left} id="in" style={inputHandleStyle} />
                 <span className="text-[10px] ml-3 text-zinc-500 dark:text-zinc-400">{t.from_dataset}</span>
             </div>
             <div className="text-[9px] text-zinc-500 mb-2">{t.connect_items}</div>
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

// --- IMAGE SECTION ---

export const ImageListNode = memo((props: NodeProps) => 
    <GenericListNode {...props} titleKey="img_list" color="bg-emerald-900" prefix="img" />
);

export const ImageConfigNode = memo(({ id, data, selected }: NodeProps) => {
    const { updateNodeData, lang } = useFlowContext();
    const t = TRANSLATIONS[lang];
    const update = (k: string, v: any) => updateNodeData(id, k, v);
    return (
        <BaseNode title={t.img_item} color="bg-emerald-800" selected={selected} id={id}>
             <div className="flex justify-start mb-2 pb-2 border-b border-zinc-200 dark:border-zinc-700">
                 <Handle type="target" position={Position.Left} id="in" style={inputHandleStyle} />
                 <span className="text-[10px] ml-3 text-zinc-500 dark:text-zinc-400">{t.from_list}</span>
             </div>
            <div className="grid grid-cols-2 gap-2">
                <div>
                    <label className={labelClass}>{t.image_name}</label>
                    <input className={inputClass} value={data.key || ""} onChange={e => update('key', e.target.value)} placeholder="train" />
                </div>
                 <div>
                    <label className={labelClass}>{t.suffix}</label>
                    <input className={inputClass} value={data.suffix || ""} onChange={e => update('suffix', e.target.value)} placeholder="_T" />
                </div>
            </div>
        </BaseNode>
    );
});

// --- TARGET SECTION ---

export const TargetListNode = memo((props: NodeProps) => 
    <GenericListNode {...props} titleKey="tgt_list" color="bg-teal-900" prefix="tgt" />
);

export const TargetConfigNode = memo(({ id, data, selected }: NodeProps) => {
    const { updateNodeData, lang } = useFlowContext();
    const t = TRANSLATIONS[lang];
    const update = (k: string, v: any) => updateNodeData(id, k, v);
    return (
        <BaseNode title={t.tgt_item} color="bg-teal-800" selected={selected} id={id}>
            <div className="flex justify-start mb-2 pb-2 border-b border-zinc-200 dark:border-zinc-700">
                 <Handle type="target" position={Position.Left} id="in" style={inputHandleStyle} />
                 <span className="text-[10px] ml-3 text-zinc-500 dark:text-zinc-400">{t.from_list}</span>
            </div>
            <div className="space-y-2">
                <div>
                    <label className={labelClass}>{t.target_name}</label>
                    <input className={inputClass} value={data.key || ""} onChange={e => update('key', e.target.value)} placeholder="train" />
                </div>
                 <div className="grid grid-cols-2 gap-2">
                    <div>
                        <label className={labelClass}>{t.image}</label>
                        <input className={inputClass} value={data.image || ""} onChange={e => update('image', e.target.value)} placeholder="train" />
                    </div>
                    <div>
                        <label className={labelClass}>{t.from_image}</label>
                        <input className={inputClass} value={data.from_image || ""} onChange={e => update('from_image', e.target.value)} />
                    </div>
                </div>
            </div>
        </BaseNode>
    );
});

// --- CAPTION SECTION ---

export const CaptionListNode = memo((props: NodeProps) => 
    <GenericListNode {...props} titleKey="cap_list" color="bg-pink-900" prefix="cap" />
);

export const CaptionConfigNode = memo(({ id, data, selected }: NodeProps) => {
    const { updateNodeData, lang } = useFlowContext();
    const t = TRANSLATIONS[lang];
    const update = (k: string, v: any) => updateNodeData(id, k, v);
    
    // Internal state for reference list check
    const toggleRef = (e: any) => {
        if(e.target.checked) update('reference_list', { reference_config: data.key, resize: "384", dropout: 0.1, min_length: 2 });
        else update('reference_list', undefined);
    };

    const updateRef = (k: string, v: any) => {
        update('reference_list', { ...data.reference_list, [k]: v });
    };

    return (
        <BaseNode title={t.cap_item} color="bg-pink-800" selected={selected} id={id}>
            <div className="flex justify-start mb-2 pb-2 border-b border-zinc-200 dark:border-zinc-700">
                 <Handle type="target" position={Position.Left} id="in" style={inputHandleStyle} />
                 <span className="text-[10px] ml-3 text-zinc-500 dark:text-zinc-400">{t.from_list}</span>
            </div>
             <div className="space-y-2">
                <div className="grid grid-cols-3 gap-1">
                    <div className="col-span-1">
                         <label className={labelClass}>{t.caption_name}</label>
                        <input className={inputClass} value={data.key || ""} onChange={e => update('key', e.target.value)} />
                    </div>
                    <div className="col-span-1">
                         <label className={labelClass}>{t.ext}</label>
                        <input className={inputClass} value={data.ext || ""} onChange={e => update('ext', e.target.value)} placeholder=".txt" />
                    </div>
                     <div className="col-span-1">
                         <label className={labelClass}>{t.image}</label>
                        <input className={inputClass} value={data.image || ""} onChange={e => update('image', e.target.value)} />
                    </div>
                </div>
                
                <div className="bg-zinc-50 dark:bg-zinc-900/30 p-2 rounded border border-zinc-200 dark:border-zinc-800">
                     <label className="text-[9px] flex items-center gap-2 text-zinc-500 dark:text-zinc-400 mb-1 cursor-pointer">
                        <input type="checkbox" className={checkboxClass} checked={!!data.reference_list} onChange={toggleRef} /> {t.include_ref}
                    </label>
                    {data.reference_list && (
                        <div className="space-y-1 pl-1 border-l border-zinc-300 dark:border-zinc-700 mt-1">
                             <div className="grid grid-cols-2 gap-1">
                                <div>
                                    <label className={labelClass}>{t.reference_name}</label>
                                    <input className={inputClass} value={data.reference_list.reference_config} onChange={e => updateRef('reference_config', e.target.value)} />
                                </div>
                                <div>
                                    <label className={labelClass}>{t.dropout}</label>
                                    <input type="number" step="0.1" className={inputClass} value={data.reference_list.dropout} onChange={e => updateRef('dropout', parseFloat(e.target.value))} />
                                </div>
                             </div>
                             <div className="grid grid-cols-2 gap-1">
                                <div>
                                    <label className={labelClass}>{t.resize}</label>
                                    <input className={inputClass} value={data.reference_list.resize} onChange={e => updateRef('resize', e.target.value)} />
                                </div>
                                <div>
                                    <label className={labelClass}>{t.min_len}</label>
                                    <input type="number" className={inputClass} value={data.reference_list.min_length} onChange={e => updateRef('min_length', parseInt(e.target.value))} />
                                </div>
                             </div>
                        </div>
                    )}
                </div>
            </div>
        </BaseNode>
    );
});

// --- BATCH SECTION ---

export const BatchListNode = memo((props: NodeProps) => 
    <GenericListNode {...props} titleKey="batch_list" color="bg-indigo-900" prefix="batch" />
);

export const BatchConfigNode = memo(({ id, data, selected }: NodeProps) => {
    const { updateNodeData, lang } = useFlowContext();
    const t = TRANSLATIONS[lang];
    const update = (k: string, v: any) => updateNodeData(id, k, v);
    return (
        <BaseNode title={t.batch_item} color="bg-indigo-800" selected={selected} id={id}>
            <div className="flex justify-start mb-2 pb-2 border-b border-zinc-200 dark:border-zinc-700">
                 <Handle type="target" position={Position.Left} id="in" style={inputHandleStyle} />
                 <span className="text-[10px] ml-3 text-zinc-500 dark:text-zinc-400">{t.from_list}</span>
            </div>
             <div className="grid grid-cols-2 gap-2">
                 <div>
                    <label className={labelClass}>{t.target}</label>
                    <input className={inputClass} value={data.target_config || ""} onChange={e => update('target_config', e.target.value)} />
                 </div>
                 <div>
                    <label className={labelClass}>{t.caption}</label>
                    <input className={inputClass} value={data.caption_config || ""} onChange={e => update('caption_config', e.target.value)} />
                 </div>
                 <div>
                    <label className={labelClass}>{t.reference}</label>
                    <input className={inputClass} value={data.reference_config || ""} onChange={e => update('reference_config', e.target.value)} />
                 </div>
                 <div>
                    <label className={labelClass}>{t.cap_dropout}</label>
                    <input type="number" step="0.1" className={inputClass} value={data.caption_dropout ?? 0.1} onChange={e => update('caption_dropout', parseFloat(e.target.value))} />
                 </div>
            </div>
        </BaseNode>
    );
});

// --- REFERENCE SECTION ---

export const ReferenceListNode = memo((props: NodeProps) => 
    <GenericListNode {...props} titleKey="ref_list" color="bg-orange-900" prefix="ref" />
);

// Individual Entry Node
export const ReferenceEntryNode = memo(({ id, data, selected }: NodeProps) => {
    const { updateNodeData, lang } = useFlowContext();
    const t = TRANSLATIONS[lang];
    const update = (k: string, v: any) => updateNodeData(id, k, v);

    return (
        <BaseNode title={t.ref_src} color="bg-orange-700" selected={selected} id={id}>
             <div className="flex justify-start mb-2 pb-2 border-b border-zinc-200 dark:border-zinc-700">
                 <Handle type="target" position={Position.Left} id="in" style={inputHandleStyle} />
                 <span className="text-[10px] ml-3 text-zinc-500 dark:text-zinc-400">{t.from_group}</span>
             </div>
            <div className="space-y-2">
                <div className="grid grid-cols-2 gap-2">
                     <div>
                        <label className={labelClass}>{t.sample_type}</label>
                        <select className={inputClass} value={data.sample_type || "from_same_name"} onChange={e => update('sample_type', e.target.value)}>
                            <option value="from_same_name">same_name</option>
                            <option value="from_subdir">subdir</option>
                        </select>
                     </div>
                     <div>
                        <label className={labelClass}>{t.image_name}</label>
                        <input className={inputClass} value={data.image || ""} onChange={e => update('image', e.target.value)} placeholder="Img Key" />
                     </div>
                </div>
                {data.sample_type === 'from_subdir' && (
                    <div className="grid grid-cols-3 gap-2">
                        <div>
                            <label className={labelClass}>{t.suffix}</label>
                            <input className={inputClass} value={data.suffix || ""} onChange={e => update('suffix', e.target.value)} />
                        </div>
                        <div>
                            <label className={labelClass}>{t.resize}</label>
                            <input className={inputClass} value={data.resize || ""} onChange={e => update('resize', e.target.value)} />
                        </div>
                        <div>
                            <label className={labelClass}>{t.count}</label>
                            <input type="number" className={inputClass} value={data.count || 0} onChange={e => update('count', parseInt(e.target.value))} />
                        </div>
                    </div>
                )}
            </div>
        </BaseNode>
    );
});

// Group (Aggregator) for Reference Entries
export const ReferenceConfigNode = memo(({ id, data, selected }: NodeProps) => {
    const { updateNodeData, lang } = useFlowContext();
    const t = TRANSLATIONS[lang];
    const update = (k: string, v: any) => updateNodeData(id, k, v);
    
    // Behaves like GenericListNode but also has a Key field
    const [slots, setSlots] = useState<string[]>(data.slots || ['entry_0']);

    useEffect(() => {
        if (!data.slots) updateNodeData(id, 'slots', slots);
    }, []);

    const updateSlots = (newSlots: string[]) => {
        setSlots(newSlots);
        updateNodeData(id, 'slots', newSlots);
    };

    const addSlot = () => updateSlots([...slots, `entry_${Math.random().toString(36).substr(2, 5)}`]);
    const removeSlot = (slot: string) => updateSlots(slots.filter(s => s !== slot));

    return (
        <BaseNode title={t.ref_item} color="bg-orange-800" selected={selected} id={id}>
             <div className="flex justify-start mb-2 pb-2 border-b border-zinc-200 dark:border-zinc-700">
                 <Handle type="target" position={Position.Left} id="in" style={inputHandleStyle} />
                 <span className="text-[10px] ml-3 text-zinc-500 dark:text-zinc-400">{t.from_list}</span>
             </div>

            <div className="mb-3">
                 <label className={labelClass}>{t.reference_name}</label>
                 <input className={inputClass} value={data.key || ""} onChange={e => update('key', e.target.value)} />
            </div>
            
             <div className="text-[9px] text-zinc-500 mb-2">{t.connect_items}</div>
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
});