import React, { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { BaseNode, inputHandleStyle } from './BaseNode';
import { OPTIMIZERS, PRECISIONS, SCHEDULERS, WEIGHTING_SCHEMES, TRANSLATIONS } from '../../constants';
import { useFlowContext } from '../FlowContext';

const inputClass = "w-full bg-white dark:bg-zinc-900 rounded border border-zinc-300 dark:border-zinc-700 px-1 text-xs text-zinc-900 dark:text-zinc-200 focus:outline-none focus:border-blue-500 transition-colors";
const labelClass = "text-[9px] text-zinc-500 font-semibold";
const checkboxClass = "appearance-none w-3.5 h-3.5 border border-zinc-400 dark:border-zinc-600 rounded bg-white dark:bg-zinc-900 checked:bg-blue-600 checked:border-blue-600 relative cursor-pointer after:content-[''] after:absolute after:hidden after:checked:block after:left-[3px] after:top-[0px] after:w-[5px] after:h-[9px] after:border-r-2 after:border-b-2 after:border-white after:rotate-45";

// --- Directory Node ---
export const DirectoryNode = memo(({ id, data, selected }: NodeProps) => {
  const { updateNodeData, lang } = useFlowContext();
  const t = TRANSLATIONS[lang];
  const update = (k: string, v: any) => updateNodeData(id, k, v);

  return (
    <BaseNode title={t.dir} color="bg-blue-900" selected={selected} id={id} data={data}>
        <div className="flex justify-start mb-4 pb-2 border-b border-zinc-200 dark:border-zinc-700">
             <Handle type="target" position={Position.Left} id="in" style={inputHandleStyle} />
             <span className="text-[10px] ml-3 text-zinc-500 dark:text-zinc-400">{t.from_script}</span>
        </div>
        <div className="space-y-2">
            <div>
                <label className={labelClass}>{t.pretrained_path}</label>
                <input className={inputClass}
                    value={data.pretrained_model_name_or_path || ''}
                    onChange={(e) => update('pretrained_model_name_or_path', e.target.value)} />
            </div>
            <div>
                <label className={labelClass}>{t.output_dir}</label>
                <input className={inputClass}
                    value={data.output_dir || ''}
                    onChange={(e) => update('output_dir', e.target.value)} />
            </div>
            <div className="grid grid-cols-2 gap-2">
                <div>
                    <label className={labelClass}>{t.logging_dir}</label>
                    <input className={inputClass}
                        value={data.logging_dir || 'logs'}
                        onChange={(e) => update('logging_dir', e.target.value)} />
                </div>
                <div>
                    <label className={labelClass}>{t.save_name}</label>
                    <input className={inputClass}
                        value={data.save_name || ''}
                        onChange={(e) => update('save_name', e.target.value)} />
                </div>
            </div>
             <div className="grid grid-cols-2 gap-2">
                 <div>
                    <label className={labelClass}>{t.global_data_dir}</label>
                    <input className={inputClass}
                        value={data.train_data_dir || ''}
                        onChange={(e) => update('train_data_dir', e.target.value)} />
                </div>
                 <div>
                    <label className={labelClass}>{t.model_path}</label>
                    <input className={inputClass}
                        value={data.model_path || ''}
                        onChange={(e) => update('model_path', e.target.value)} placeholder="Separate model" />
                </div>
            </div>
            <div>
                <label className={labelClass}>{t.vae_path}</label>
                <input className={inputClass}
                    value={data.vae_path || ''}
                    onChange={(e) => update('vae_path', e.target.value)} placeholder="Separate VAE" />
            </div>
        </div>
    </BaseNode>
  );
});

// --- LoRA Node ---
export const LoRANode = memo(({ id, data, selected }: NodeProps) => {
    const { updateNodeData, lang } = useFlowContext();
    const t = TRANSLATIONS[lang];
    const update = (k: string, v: any) => updateNodeData(id, k, v);
  
    return (
      <BaseNode title={t.lora} color="bg-purple-900" selected={selected} id={id} data={data}>
           <div className="flex justify-start mb-4 pb-2 border-b border-zinc-200 dark:border-zinc-700">
             <Handle type="target" position={Position.Left} id="in" style={inputHandleStyle} />
             <span className="text-[10px] ml-3 text-zinc-500 dark:text-zinc-400">{t.from_script}</span>
          </div>
          <div className="space-y-2">
              <div className="grid grid-cols-2 gap-2">
                <div>
                    <label className={labelClass}>{t.rank}</label>
                    <input type="number" className={inputClass}
                        value={data.rank ?? 16} onChange={(e) => update('rank', parseInt(e.target.value))} />
                </div>
                <div>
                    <label className={labelClass}>{t.alpha}</label>
                    <input type="number" className={inputClass}
                        value={data.rank_alpha ?? 16} onChange={(e) => update('rank_alpha', parseInt(e.target.value))} />
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-2 border-t border-zinc-200 dark:border-zinc-700/50 pt-2 items-center">
                 <div className="flex items-center gap-2 h-full pt-3">
                    <input type="checkbox" className={checkboxClass} checked={data.use_lokr || false} onChange={e => update('use_lokr', e.target.checked)} />
                    <label className="text-[10px] text-zinc-700 dark:text-zinc-300">{t.use_lokr}</label>
                 </div>
                 
                 <div>
                    <label className={labelClass}>{t.lokr_factor}</label>
                    <input type="number" className={inputClass}
                        value={data.lokr_factor ?? 4} onChange={(e) => update('lokr_factor', parseInt(e.target.value))} />
                 </div>
              </div>

              <div>
                 <label className={labelClass}>{t.lora_layers}</label>
                 <input className={inputClass}
                    value={data.lora_layers || ''} placeholder="all or layers..." onChange={(e) => update('lora_layers', e.target.value)} />
              </div>
          </div>
      </BaseNode>
    );
});

// --- Misc Node ---
export const MiscNode = memo(({ id, data, selected }: NodeProps) => {
    const { updateNodeData, lang } = useFlowContext();
    const t = TRANSLATIONS[lang];
    const update = (k: string, v: any) => updateNodeData(id, k, v);
  
    return (
      <BaseNode title={t.misc} color="bg-slate-800" selected={selected} id={id} data={data}>
           <div className="flex justify-start mb-4 pb-2 border-b border-zinc-200 dark:border-zinc-700">
             <Handle type="target" position={Position.Left} id="in" style={inputHandleStyle} />
             <span className="text-[10px] ml-3 text-zinc-500 dark:text-zinc-400">{t.from_script}</span>
          </div>
          
          <div className="space-y-3">
              {/* Basic Settings */}
              <div className="grid grid-cols-2 gap-2">
                  <div className="col-span-2">
                      <label className={labelClass}>{t.resolution}</label>
                      <input className={inputClass}
                          value={data.resolution || "1024"} onChange={(e) => update('resolution', e.target.value)} />
                  </div>
                   <div>
                      <label className={labelClass}>{t.batch_size}</label>
                      <input type="number" className={inputClass}
                          value={data.train_batch_size ?? 1} onChange={(e) => update('train_batch_size', parseInt(e.target.value))} />
                  </div>
                  <div>
                      <label className={labelClass}>{t.epochs}</label>
                      <input type="number" className={inputClass}
                          value={data.num_train_epochs ?? 10} onChange={(e) => update('num_train_epochs', parseInt(e.target.value))} />
                  </div>
              </div>
              
              <div className="grid grid-cols-2 gap-2">
                  <div>
                      <label className={labelClass}>{t.learning_rate}</label>
                      <input type="number" className={inputClass}
                          value={data.learning_rate ?? 1e-4} onChange={(e) => update('learning_rate', parseFloat(e.target.value))} />
                  </div>
                   <div>
                      <label className={labelClass}>{t.optimizer}</label>
                      <select className={inputClass}
                          value={data.optimizer || 'adamw'} onChange={(e) => update('optimizer', e.target.value)}>
                          {OPTIMIZERS.map(o => <option key={o} value={o}>{o}</option>)}
                      </select>
                  </div>
              </div>
              
              <div className="grid grid-cols-2 gap-2">
                   <div>
                      <label className={labelClass}>{t.precision}</label>
                      <select className={inputClass}
                          value={data.mixed_precision || 'bf16'} onChange={(e) => update('mixed_precision', e.target.value)}>
                          {PRECISIONS.map(p => <option key={p} value={p}>{p}</option>)}
                      </select>
                  </div>
                  <div>
                      <label className={labelClass}>{t.scheduler}</label>
                       <select className={inputClass}
                          value={data.lr_scheduler || 'cosine'} onChange={(e) => update('lr_scheduler', e.target.value)}>
                          {SCHEDULERS.map(s => <option key={s} value={s}>{s}</option>)}
                      </select>
                  </div>
              </div>
              
              <div className="grid grid-cols-2 gap-2">
                   <div>
                      <label className={labelClass}>{t.seed}</label>
                      <input type="number" className={inputClass}
                          value={data.seed ?? 42} onChange={(e) => update('seed', parseInt(e.target.value))} />
                  </div>
                   <div>
                      <label className={labelClass}>{t.cap_dropout}</label>
                      <input type="number" step="0.1" className={inputClass}
                          value={data.caption_dropout ?? 0.1} onChange={(e) => update('caption_dropout', parseFloat(e.target.value))} />
                  </div>
              </div>
              
              <div className="grid grid-cols-2 gap-2">
                   <div>
                      <label className={labelClass}>{t.save_every}</label>
                      <input type="number" className={inputClass}
                          value={data.save_model_epochs ?? 1} onChange={(e) => update('save_model_epochs', parseInt(e.target.value))} />
                  </div>
                  <div>
                      <label className={labelClass}>{t.blocks_to_swap}</label>
                      <input type="number" className={inputClass}
                          value={data.blocks_to_swap ?? 0} onChange={(e) => update('blocks_to_swap', parseInt(e.target.value))} />
                  </div>
              </div>

              {/* Toggles */}
              <div className="grid grid-cols-2 gap-2 pt-1">
                  <div className="flex items-center gap-2 p-1.5 rounded bg-zinc-50 dark:bg-zinc-900/40 border border-zinc-200 dark:border-zinc-700/50">
                      <input type="checkbox" className={checkboxClass} checked={data.gradient_checkpointing || false} onChange={e => update('gradient_checkpointing', e.target.checked)} />
                      <label className="text-[9px] text-zinc-700 dark:text-zinc-300 font-semibold">{t.grad_chkpt}</label>
                  </div>
                  <div className="flex items-center gap-2 p-1.5 rounded bg-zinc-50 dark:bg-zinc-900/40 border border-zinc-200 dark:border-zinc-700/50">
                      <input type="checkbox" className={checkboxClass} checked={data.recreate_cache || false} onChange={e => update('recreate_cache', e.target.checked)} />
                      <label className="text-[9px] text-zinc-700 dark:text-zinc-300 font-semibold">{t.recr_cache}</label>
                  </div>
              </div>

              <div className="h-px bg-zinc-200 dark:bg-zinc-700 my-1"></div>

              {/* Flattened Advanced Settings */}
               <div className="grid grid-cols-2 gap-2">
                    <div>
                        <label className={labelClass}>{t.val_epochs}</label>
                        <input type="number" className={inputClass}
                            value={data.validation_epochs ?? 1} onChange={(e) => update('validation_epochs', parseInt(e.target.value))} />
                    </div>
                    <div>
                        <label className={labelClass}>{t.val_ratio}</label>
                        <input type="number" step="0.05" className={inputClass}
                            value={data.validation_ratio ?? 0.1} onChange={(e) => update('validation_ratio', parseFloat(e.target.value))} />
                    </div>
               </div>
               
                <div>
                    <label className={labelClass}>{t.weighting_scheme}</label>
                    <select className={inputClass}
                        value={data.weighting_scheme || 'logit_normal'} onChange={(e) => update('weighting_scheme', e.target.value)}>
                        {WEIGHTING_SCHEMES.map(s => <option key={s} value={s}>{s}</option>)}
                    </select>
                </div>
                
                <div className="grid grid-cols-3 gap-1">
                     <div>
                        <label className={labelClass}>{t.logit_mean}</label>
                        <input type="number" className={inputClass}
                            value={data.logit_mean ?? 0.0} onChange={(e) => update('logit_mean', parseFloat(e.target.value))} />
                    </div>
                     <div>
                        <label className={labelClass}>{t.logit_std}</label>
                        <input type="number" className={inputClass}
                            value={data.logit_std ?? 1.0} onChange={(e) => update('logit_std', parseFloat(e.target.value))} />
                    </div>
                    <div>
                        <label className={labelClass}>{t.mode_scale}</label>
                        <input type="number" className={inputClass}
                            value={data.mode_scale ?? 1.29} onChange={(e) => update('mode_scale', parseFloat(e.target.value))} />
                    </div>
                </div>

                 <div className="grid grid-cols-2 gap-2">
                     <div>
                        <label className={labelClass}>{t.guidance_scale}</label>
                        <input type="number" className={inputClass}
                            value={data.guidance_scale ?? 1.0} onChange={(e) => update('guidance_scale', parseFloat(e.target.value))} />
                    </div>
                    <div>
                        <label className={labelClass}>{t.mask_dropout}</label>
                        <input type="number" className={inputClass}
                            value={data.mask_dropout ?? 0.01} onChange={(e) => update('mask_dropout', parseFloat(e.target.value))} />
                    </div>
                 </div>
                 

                 
                 <div>
                     <label className={labelClass}>{t.freeze_transformers}</label>
                     <input className={`${inputClass} mb-1`}
                        placeholder="e.g. 5,7,10" value={data.freeze_transformer_layers || ''} onChange={(e) => update('freeze_transformer_layers', e.target.value)} />
                     <input className={inputClass}
                        placeholder="Single layers..." value={data.freeze_single_transformer_layers || ''} onChange={(e) => update('freeze_single_transformer_layers', e.target.value)} />
                 </div>
                 
                 <div>
                     <label className={labelClass}>{t.report_to}</label>
                     <select className={inputClass}
                        value={data.report_to || 'tensorboard'} onChange={(e) => update('report_to', e.target.value)}>
                        <option value="tensorboard">tensorboard</option>
                        <option value="wandb">wandb</option>
                        <option value="none">none</option>
                    </select>
                 </div>
                 
                 <div>
                    <label className={labelClass}>{t.resume_from}</label>
                    <input className={inputClass}
                        placeholder="Path to saved directory..." value={data.resume_from || ''} onChange={(e) => update('resume_from', e.target.value)} />
                </div>
          </div>
      </BaseNode>
    );
});

// --- Config Template Node ---
export const ConfigTemplateNode = memo(({ id, data, selected }: NodeProps) => {
    const { updateNodeData, lang } = useFlowContext();
    const t = TRANSLATIONS[lang];
    const update = (k: string, v: any) => updateNodeData(id, k, v);

    return (
        <BaseNode title={t.template} color="bg-amber-800" selected={selected} id={id} data={data}>
             <div className="flex justify-start mb-2 pb-2 border-b border-zinc-200 dark:border-zinc-700">
                 <Handle type="target" position={Position.Left} id="in" style={inputHandleStyle} />
                 <span className="text-[10px] ml-3 text-zinc-500 dark:text-zinc-400">{t.from_script}</span>
             </div>
            <div className="text-xs text-zinc-500 dark:text-zinc-400 mb-2">
                {t.paste_template}
            </div>
            <textarea 
                className="w-full h-24 bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-zinc-700 rounded p-1 text-[9px] font-mono text-zinc-900 dark:text-zinc-300 focus:outline-none focus:border-amber-600 transition-colors"
                placeholder='{"learning_rate": 0.001 ...}'
                value={data.templateJson || ''}
                onChange={e => update('templateJson', e.target.value)}
            />
        </BaseNode>
    )
})