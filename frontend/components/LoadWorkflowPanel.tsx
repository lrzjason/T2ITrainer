import React, { useState, useEffect } from 'react';
import { X, Save, FilePlus, Trash2 } from 'lucide-react';
import { TRANSLATIONS } from '../constants';
import { Language } from './FlowContext';

interface WorkflowTemplate {
  id: string;
  name: string;
  path: string;
}

interface LoadWorkflowPanelProps {
  isOpen: boolean;
  onClose: () => void;
  onLoadWorkflow: (workflow: any) => void;
  onSaveWorkflow: (name: string, workflow: any) => Promise<void>;
  onSaveCurrentWorkflow?: (workflow: any) => Promise<void>;
  onLoadCurrentWorkflow?: () => Promise<void>;
  onDeleteTemplate: (path: string) => Promise<void>;
  currentWorkflow: { nodes: any[]; edges: any[] };
  setNodes: (nodes: any[]) => void;
  setEdges: (edges: any[]) => void;
  lang: 'en' | 'cn';
}

export const LoadWorkflowPanel: React.FC<LoadWorkflowPanelProps> = ({
  isOpen,
  onClose,
  onLoadWorkflow,
  onSaveWorkflow,
  onSaveCurrentWorkflow,
  onLoadCurrentWorkflow,
  onDeleteTemplate,
  currentWorkflow,
  setNodes,
  setEdges,
  lang
}) => {
  const t = (key: keyof typeof TRANSLATIONS.en) => TRANSLATIONS[lang][key] || key;
  
  const [templates, setTemplates] = useState<WorkflowTemplate[]>([]);
  const [filteredTemplates, setFilteredTemplates] = useState<WorkflowTemplate[]>([]);
  const [selectedTemplate, setSelectedTemplate] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');

  // Filter templates when searchTerm changes
  useEffect(() => {
    if (searchTerm) {
      const filtered = templates.filter(template => 
        template.name.toLowerCase().includes(searchTerm.toLowerCase()) || 
        template.path.toLowerCase().includes(searchTerm.toLowerCase())
      );
      setFilteredTemplates(filtered);
    } else {
      setFilteredTemplates(templates);
    }
  }, [searchTerm, templates]);

  // Load templates from the templates directory
  useEffect(() => {
    if (isOpen) {
      loadTemplates();
    }
  }, [isOpen]);

  // Listen for template refresh events
  useEffect(() => {
    const handleRefreshTemplates = () => {
      loadTemplates();
    };

    window.addEventListener('refreshTemplates', handleRefreshTemplates);
    
    return () => {
      window.removeEventListener('refreshTemplates', handleRefreshTemplates);
    };
  }, []);

  const loadTemplates = async () => {
    try {
      setLoading(true);
      
      // Load built-in templates
      const builtInTemplates: WorkflowTemplate[] = [
        { id: '1', name: 'Single Image Workflow', path: '/templates/single.json' },
        { id: '2', name: 'Pair Comparison Workflow', path: '/templates/pairs.json' },
        { id: '3', name: 'Multiple References Workflow', path: '/templates/multiple.json' },
        { id: '4', name: 'Mixed Layout Workflow', path: '/templates/mixed.json' }
      ];
      
      // Load custom templates from backend
      const response = await fetch('/api/templates/custom');
      const data = await response.json();
      const customTemplates: WorkflowTemplate[] = data.templates.map((name: string, index: number) => ({
        id: `custom_${index}`,
        name: name.replace(/\.json$/, ''),
        path: `/custom_templates/${name}`
      }));
      
      // Combine built-in and custom templates
      setTemplates([...builtInTemplates, ...customTemplates]);
      setError(null);
    } catch (err) {
      setError('Failed to load templates');
      console.error('Error loading templates:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleLoadTemplate = async (templatePath: string) => {
    try {
      setLoading(true);
      
      // Check if this is a custom template
      if (templatePath.startsWith('/custom_templates/')) {
        // For custom templates, we need to fetch them differently
        // Extract the template name from the path
        const templateName = templatePath.split('/').pop();
        if (!templateName) {
          throw new Error('Invalid template path');
        }
        
        // Fetch the template from the backend
        const response = await fetch(`/custom_templates/${templateName}`);
        if (!response.ok) {
          throw new Error(`Failed to load template: ${response.statusText}`);
        }
        const workflow = await response.json();
        onLoadWorkflow(workflow);
      } else {
        // Load built-in template from public directory
        const response = await fetch(templatePath);
        if (!response.ok) {
          throw new Error(`Failed to load template: ${response.statusText}`);
        }
        const workflow = await response.json();
        onLoadWorkflow(workflow);
      }
    } catch (err) {
      setError('Failed to load workflow template');
      console.error('Error loading template:', err);
    } finally {
      setLoading(false);
    }
  };



  const handleDeleteTemplate = async (templatePath: string, templateName: string) => {
    if (!window.confirm(t('confirm_delete_template').replace('{templateName}', templateName))) {
      return;
    }

    try {
      setLoading(true);
      await onDeleteTemplate(templatePath);
      await loadTemplates(); // Refresh the template list
    } catch (err) {
      setError('Failed to delete template');
      console.error('Error deleting template:', err);
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="absolute left-20 top-4 bottom-4 w-[500px] bg-white/20 dark:bg-zinc-900/50 backdrop-blur-sm border border-zinc-200 dark:border-zinc-800 rounded-2xl shadow-2xl flex flex-col z-40 animate-in slide-in-from-left-4 fade-in duration-200">
      <div className="p-4 border-b border-zinc-200 dark:border-zinc-800 flex justify-between items-center bg-white/70 dark:bg-zinc-900/70 rounded-t-2xl">
        <h2 className="text-zinc-600 dark:text-zinc-400 text-xs font-bold uppercase tracking-wider flex items-center gap-2">
          {t('load')}
        </h2>
        <button onClick={onClose} className="text-zinc-400 hover:text-zinc-900 dark:hover:text-white">
          <X size={14}/>
        </button>
      </div>
      
      <div className="flex-1 overflow-hidden p-5 flex flex-col gap-6">
        {/* Current Workflow Display */}
        <div className="mb-4">
          <div className="flex justify-between items-center mb-2">
            <h3 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">
              {t('current_workflow')}
            </h3>
            <button
              onClick={() => {
                if (onSaveCurrentWorkflow) {
                  // Use the current workflow save function if available
                  onSaveCurrentWorkflow(currentWorkflow);
                } else {
                  // Fallback to regular save as template
                  const name = prompt(t('enter_template_name'));
                  if (name) {
                    onSaveWorkflow(name.endsWith('.json') ? name : `${name}.json`, currentWorkflow);
                  }
                }
              }}
              className="px-2 py-1 bg-green-500 hover:bg-green-600 text-white text-xs rounded transition-colors"
            >
              {t('save_current')}
            </button>
          </div>
          <div className="flex items-center justify-between p-2 rounded-lg border bg-white dark:bg-zinc-900 border-zinc-200 dark:border-zinc-800">
            <div className="flex-1">
              <div className="font-medium text-zinc-900 dark:text-white text-sm">
                {t('current_workflow')}
              </div>
              <div className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
                /current/workflow.json
              </div>
            </div>
            <div className="flex items-center gap-1">
              <button
                onClick={async () => {
                  if (onLoadCurrentWorkflow) {
                    try {
                      await onLoadCurrentWorkflow();
                    } catch (error) {
                      console.error('Error loading current workflow:', error);
                    }
                  }
                }}
                className="px-2 py-1 bg-blue-500 hover:bg-blue-600 text-white text-xs rounded transition-colors"
                title="Load current workflow"
              >
                {t('load_template')}
              </button>
              <button
                onClick={() => {
                  if (window.confirm(t('confirm_clear_nodes'))) {
                    // Clear all nodes and edges
                    setNodes([]);
                    setEdges([]);
                  }
                }}
                className="p-1 text-zinc-500 hover:text-red-500 dark:text-zinc-400 hover:dark:text-red-400 transition-colors"
                title="Clear all nodes"
              >
                <Trash2 size={14} />
              </button>
            </div>
          </div>
        </div>

        {/* Available Templates */}
        <div className="mb-4 flex-1 flex flex-col min-h-0">
          <div className="flex justify-between items-center mb-2">
            <h3 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">
              {t('available_templates')}
            </h3>
            <button
              onClick={() => {
                const name = prompt(t('enter_template_name'));
                if (name) {
                  onSaveWorkflow(name.endsWith('.json') ? name : `${name}.json`, currentWorkflow);
                }
              }}
              className="px-2 py-1 bg-green-500 hover:bg-green-600 text-white text-xs rounded transition-colors"
            >
              {t('save_as_template')}
            </button>
          </div>
          
          {/* Search Input */}
          <div className="mb-3">
            <input
              type="text"
              placeholder={t('search_templates')}
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-900 dark:text-zinc-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>
          
          {loading && filteredTemplates.length === 0 ? (
            <div className="text-center py-4 text-zinc-500 dark:text-zinc-400">
              {t('loading_templates')}
            </div>
          ) : filteredTemplates.length === 0 ? (
            <div className="text-center py-4 text-zinc-500 dark:text-zinc-400">
              {t('no_templates_found')}
            </div>
          ) : (
            <div className="flex-1 overflow-y-auto space-y-1">
              {filteredTemplates.map((template) => (
                <div
                  key={template.id}
                  className={`flex items-center justify-between p-2 rounded-lg border transition-colors ${
                    selectedTemplate === template.id
                      ? 'bg-blue-50 dark:bg-blue-900/30 border-blue-300 dark:border-blue-700'
                      : 'bg-white dark:bg-zinc-900 border-zinc-200 dark:border-zinc-800 hover:bg-zinc-50 dark:hover:bg-zinc-800'
                  }`}
                >
                  <div 
                    className="flex-1 cursor-pointer"
                    onClick={() => setSelectedTemplate(template.id)}
                  >
                    <div className="font-medium text-zinc-900 dark:text-white text-sm">
                      {template.name}
                    </div>
                    <div className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
                      {template.path}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => handleLoadTemplate(template.path)}
                      className="px-2 py-1 bg-blue-500 hover:bg-blue-600 text-white text-xs rounded transition-colors"
                    >
                      {t('load_template')}
                    </button>
                    <button
                      onClick={() => handleDeleteTemplate(template.path, template.name)}
                      className="p-1 text-zinc-500 hover:text-red-500 dark:text-zinc-400 hover:dark:text-red-400 transition-colors"
                      title={t('delete_template')}
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>


      </div>
    </div>
  );
};