import React, { useState } from 'react';
import { X, Github, Info, Terminal, Layers, Component, Box } from 'lucide-react';
import { TRANSLATIONS } from '../constants';
import packageJson from '../package.json';

interface AboutPanelProps {
  isOpen: boolean;
  onClose: () => void;
  lang: 'en' | 'zh';
}

export const AboutPanel: React.FC<AboutPanelProps> = ({
  isOpen,
  onClose,
  lang
}) => {
  const t = (key: keyof typeof TRANSLATIONS.en) => TRANSLATIONS[lang][key] || key;
  
  const [activeTab, setActiveTab] = useState<'about' | 'docs'>('about');

  const DocSection = ({ title, icon: Icon, children }: any) => (
    <div className="mb-6">
      <h4 className="text-base font-bold text-zinc-900 dark:text-zinc-100 mb-3 flex items-center gap-2 border-b border-zinc-200 dark:border-zinc-800 pb-1">
        <Icon size={18} className="text-blue-500"/> {title}
      </h4>
      <div className="space-y-4">
        {children}
      </div>
    </div>
  );

  const NodeDoc = ({ title, inputs, fields, desc }: any) => (
    <div className="bg-zinc-50 dark:bg-zinc-900/50 p-3 rounded-lg border border-zinc-100 dark:border-zinc-800">
      <h5 className="font-bold text-zinc-800 dark:text-zinc-200 text-sm mb-1">{title}</h5>
      {desc && <p className="text-xs text-zinc-500 mb-2 italic">{desc}</p>}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
        {inputs && inputs.length > 0 && (
          <div>
            <span className="font-semibold text-emerald-600 dark:text-emerald-400">Inputs:</span> 
            <span className="text-zinc-600 dark:text-zinc-400 ml-1">{inputs.join(', ')}</span>
          </div>
        )}
        {fields && fields.length > 0 && (
          <div>
            <span className="font-semibold text-blue-600 dark:text-blue-400">Fields:</span>
            <span className="text-zinc-600 dark:text-zinc-400 ml-1">{fields.join(', ')}</span>
          </div>
        )}
      </div>
    </div>
  );

  if (!isOpen) return null;

  return (
    <div className="absolute left-20 top-2 bottom-2 w-[500px] bg-white/20 dark:bg-zinc-900/50 backdrop-blur-sm border border-zinc-200 dark:border-zinc-800 rounded-2xl shadow-2xl flex flex-col z-40 animate-in slide-in-from-left-4 fade-in duration-200">
      <div className="p-3 border-b border-zinc-200 dark:border-zinc-800 flex justify-between items-center bg-white/70 dark:bg-zinc-900/70 rounded-t-2xl">
        <h2 className="text-zinc-600 dark:text-zinc-400 text-xs font-bold uppercase tracking-wider">
          {t('about')}
        </h2>
        <button onClick={onClose} className="text-zinc-400 hover:text-zinc-900 dark:hover:text-white">
          <X size={14}/>
        </button>
      </div>
      
      <div className="flex border-b border-zinc-200 dark:border-zinc-800 bg-white/50 dark:bg-zinc-900/50 backdrop-blur-md">
        <button 
          onClick={() => setActiveTab('about')}
          className={`flex-1 py-2.5 text-xs font-bold uppercase tracking-wider transition-all ${activeTab === 'about' ? 'bg-zinc-50 dark:bg-zinc-900/50 text-blue-600 dark:text-blue-400 border-b-2 border-blue-500' : 'text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200 hover:bg-zinc-50/50 dark:hover:bg-zinc-800/50'}`}
        >
          {t('about_tab')}
        </button>
        <button 
          onClick={() => setActiveTab('docs')}
          className={`flex-1 py-2.5 text-xs font-bold uppercase tracking-wider transition-all ${activeTab === 'docs' ? 'bg-zinc-50 dark:bg-zinc-900/50 text-emerald-600 dark:text-emerald-400 border-b-2 border-emerald-500' : 'text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200 hover:bg-zinc-50/50 dark:hover:bg-zinc-800/50'}`}
        >
          {t('docs_tab')}
        </button>
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-hidden overflow-y-auto scrollbar-thin py-3 px-4 flex flex-col gap-4">
        {activeTab === 'about' ? (
          <div className="space-y-6 text-zinc-700 dark:text-zinc-300">
            {/* Repo Info and Version */}
            <section className="text-center pt-1">
              <a href="https://github.com/lrzjason/T2ITrainer" target="_blank" rel="noreferrer" className="text-lg font-bold text-zinc-900 dark:text-white flex items-center justify-center gap-2 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
                <Github size={20}/> {t('t2itrainer_repository')}
              </a>
              <div className="inline-flex items-center gap-2 bg-zinc-100 dark:bg-zinc-800 px-3 py-1 rounded-full text-xs font-medium text-zinc-600 dark:text-zinc-300">
                <span>Frontend Version</span>
                <span className="bg-blue-500 text-white px-2 py-0.5 rounded-full text-[10px] font-bold">V{packageJson.version}</span>
              </div>
            </section>

            {/* Contact Info */}
            <section>
              <h3 className="text-base font-bold text-zinc-900 dark:text-white mb-3 flex items-center gap-2">
                📬 {t('contact_information')}
              </h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                <div className="bg-zinc-50 dark:bg-zinc-900 p-2.5 rounded-xl border border-zinc-100 dark:border-zinc-800 flex flex-col">
                  <span className="font-bold text-zinc-500 text-[10px] uppercase mb-1">{t('twitter')}</span>
                  <a href="https://twitter.com/Lrzjason" target="_blank" rel="noreferrer" className="text-blue-500 text-sm font-medium hover:underline truncate">@Lrzjason</a>
                </div>
                <div className="bg-zinc-50 dark:bg-zinc-900 p-2.5 rounded-xl border border-zinc-100 dark:border-zinc-800 flex flex-col">
                  <span className="font-bold text-zinc-500 text-[10px] uppercase mb-1">{t('email')}</span>
                  <a href="mailto:lrzjason@gmail.com" className="text-blue-500 text-sm font-medium hover:underline truncate">lrzjason@gmail.com</a>
                </div>
                <div className="bg-zinc-50 dark:bg-zinc-900 p-2.5 rounded-xl border border-zinc-100 dark:border-zinc-800 flex flex-col">
                  <span className="font-bold text-zinc-500 text-[10px] uppercase mb-1">{t('qq_group')}</span>
                  <span className="font-mono text-sm truncate">866612947</span>
                </div>
                <div className="bg-zinc-50 dark:bg-zinc-900 p-2.5 rounded-xl border border-zinc-100 dark:border-zinc-800 flex flex-col">
                  <span className="font-bold text-zinc-500 text-[10px] uppercase mb-1">{t('wechat_id')}</span>
                  <span className="font-mono text-sm truncate">fkdeai</span>
                </div>
                <div className="bg-zinc-50 dark:bg-zinc-900 p-2.5 rounded-xl border border-zinc-100 dark:border-zinc-800 flex flex-col">
                  <span className="font-bold text-zinc-500 text-[10px] uppercase mb-1">{t('civitai')}</span>
                  <a href="https://civitai.com/user/xiaozhijason" target="_blank" rel="noreferrer" className="text-blue-500 text-sm font-medium hover:underline truncate">xiaozhijason</a>
                </div>
                <div className="bg-zinc-50 dark:bg-zinc-900 p-2.5 rounded-xl border border-zinc-100 dark:border-zinc-800 flex flex-col">
                  <span className="font-bold text-zinc-500 text-[10px] uppercase mb-1">{t('bilibili')}</span>
                  <a href="https://space.bilibili.com/443010" target="_blank" rel="noreferrer" className="text-blue-500 text-sm font-medium hover:underline truncate">小志Jason</a>
                </div>
              </div>
            </section>

            {/* Sponsor Info */}
            <section>
              <h3 className="text-base font-bold text-zinc-900 dark:text-white mb-3 text-center">
                ❤️ {t('sponsor_message')}
              </h3>
              <div className="flex justify-center bg-zinc-50 dark:bg-zinc-900/30 p-3 rounded-xl ">
                <div className="text-center group px-1">
                  <div className="bg-white p-1.5 rounded-lg shadow-sm mb-1.5 group-hover:shadow-md transition-shadow">
                    <img src="/images/bmc_qr.png" alt="Buy Me a Coffee QR" className="w-40 h-40 object-contain mix-blend-multiply dark:mix-blend-normal" />
                  </div>
                  <p className="font-bold text-[10px] uppercase tracking-wide text-zinc-500">{t('buy_me_a_coffee')}</p>
                </div>
                <div className="text-center group">
                  <div className="bg-white p-1.5 rounded-lg shadow-sm mb-1.5 group-hover:shadow-md transition-shadow">
                    <img src="/images/wechat.jpg" alt="WeChat QR" className="w-40 h-40 object-contain mix-blend-multiply dark:mix-blend-normal" />
                  </div>
                  <p className="font-bold text-[10px] uppercase tracking-wide text-zinc-500">{t('wechat')}</p>
                </div>
              </div>
            </section>
          </div>
        ) : (
          <div className="pb-4">
            <p className="mb-3 text-xs text-zinc-500 dark:text-zinc-400 bg-blue-50 dark:bg-blue-900/20 p-2.5 rounded border border-blue-100 dark:border-blue-900/50">
              <Info size={13} className="inline mr-1 text-blue-500 mb-0.5"/>
              {t('documentation_description')}
            </p>
            
            <div className="space-y-4">
              <DocSection title={t('core_nodes')} icon={Terminal}>
                <NodeDoc 
                  title="Script Loader"
                  desc="The main entry point for generating the training command and configuration file."
                  fields={['Python Script', 'Config Path (Output)']}
                  inputs={['Connect to: Directory Setup, LoRA Config, Misc Settings, Dataset List, Old Schema Root']}
                />
                <NodeDoc 
                  title="Directory Setup"
                  desc="Defines global paths for models and output data."
                  inputs={['From Script']}
                  fields={['Pretrained Model Path', 'Output Dir', 'Logging Dir', 'Save Name', 'Global Data Dir', 'Model Path', 'VAE Path']}
                />
                <NodeDoc 
                  title="LoRA Config"
                  desc="Network configuration for LoRA training."
                  inputs={['From Script']}
                  fields={['Rank', 'Alpha', 'Use LoKR', 'LoKR Factor', 'LoRA Layers']}
                />
                <NodeDoc 
                  title="Misc Settings"
                  desc="General hyperparameters and advanced training settings."
                  inputs={['From Script']}
                  fields={['Resolution', 'Batch Size', 'Epochs', 'Learning Rate', 'Optimizer', 'Precision', 'Scheduler', 'Seed', 'Caption Dropout', 'Save Every N Epochs', 'Blocks to Swap', 'Gradient Checkpointing', 'Recreate Cache', 'Validation Epochs', 'Validation Ratio', 'Weighting Scheme']}
                />
              </DocSection>

              <DocSection title={t('structure_nodes')} icon={Layers}>
                <NodeDoc 
                  title="Dataset List"
                  desc="Aggregates multiple dataset configurations."
                  inputs={['From Script']}
                  fields={['Add/Remove Dataset Slots']}
                />
                <NodeDoc 
                  title="Dataset Configuration"
                  desc="Configuration for a specific dataset block. Acts as a hub for component lists."
                  inputs={['From Dataset List']}
                  fields={['Data Directory', 'Repeats', 'Resolution', 'Recreate Cache (Target/Ref/Cap)']}
                />
              </DocSection>

              <DocSection title={t('component_nodes')} icon={Component}>
                <NodeDoc 
                  title="Lists (Image, Target, Caption, Batch, Reference)"
                  desc="Intermediate nodes that manage lists of specific items. Connects Dataset Config to individual Items."
                  inputs={['From Dataset Config']}
                  fields={['Add/Remove Slots']}
                />
                <NodeDoc 
                  title="Image Item"
                  desc="Defines image processing rules."
                  inputs={['From Image List']}
                  fields={['Image Name (Key)', 'Suffix']}
                />
                <NodeDoc 
                  title="Target Item"
                  desc="Defines target image pairs for training."
                  inputs={['From Target List']}
                  fields={['Target Name (Key)', 'Image', 'From Image']}
                />
                <NodeDoc 
                  title="Reference Group"
                  desc="Aggregates multiple reference sources under a single key."
                  inputs={['From Reference List']}
                  fields={['Reference Name (Key)']}
                />
                <NodeDoc 
                  title="Reference Source"
                  desc="A specific source of reference images."
                  inputs={['From Reference Group']}
                  fields={['Sample Type (same_name/subdir)', 'Image Name', 'Suffix', 'Resize', 'Count']}
                />
                <NodeDoc 
                  title="Caption Item"
                  desc="Defines captioning rules and optional reference lists."
                  inputs={['From Caption List']}
                  fields={['Caption Name (Key)', 'Extension', 'Image', 'Include Reference List (Ref Name, Dropout, Resize, Min Length)']}
                />
                <NodeDoc 
                  title="Batch Item"
                  desc="Links Target, Caption, and Reference configs for batching."
                  inputs={['From Batch List']}
                  fields={['Target Config', 'Caption Config', 'Reference Config', 'Caption Dropout']}
                />
              </DocSection>
              
              <DocSection title={t('old_schema_nodes')} icon={Box}>
                <NodeDoc 
                  title="Old Schema Root"
                  desc="Root node for the legacy configuration schema."
                  inputs={['From Script']}
                  fields={['Connects to: Old Image List, Old Caption List, Old Training Set List']}
                />
                <NodeDoc 
                  title="Old Image/Caption/Ref Items"
                  desc="Legacy configuration items."
                  inputs={['From respective Lists']}
                  fields={['Key', 'Suffix', 'Extension', 'Instruction', 'Target', 'Dropout', 'Resize']}
                />
              </DocSection>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};