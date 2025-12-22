import React, { useCallback, useState, useEffect, useRef } from 'react';
import ReactFlow, {
  addEdge,
  Background,
  Connection,
  Node,
  Edge,
  applyNodeChanges,
  applyEdgeChanges,
  NodeChange,
  EdgeChange,
  OnNodesChange,
  OnEdgesChange,
  SelectionMode,
  useReactFlow,
  OnConnectStartParams
} from 'reactflow';
import { LoadWorkflowPanel } from './LoadWorkflowPanel';
import { AboutPanel } from './AboutPanel';
import {
    LayoutGrid,
    Terminal,
    Save,
    FolderOpen,
    X,
    Moon,
    Sun,
    Info,
    Github,
    Box,
    Layers,
    Component,
    Settings,
    Play
} from 'lucide-react';

import { connectTrainingWebSocket, sendTrainingConfig, disconnectTrainingWebSocket, stopTraining, getWebSocketStatus, enableTrainingWebSocketDebug, disableTrainingWebSocketDebug } from '../utils/api';
import { ScriptNode } from './nodes/ScriptNode';
import { ChatPanel } from './ChatPanel';
import { DirectoryNode, LoRANode, MiscNode, ConfigTemplateNode } from './nodes/GlobalSectionNodes';
import {
    OldSchemaRootNode,
    OldImageListNode, OldImageItemNode,
    OldCaptionListNode, OldCaptionItemNode, OldRefImageListNode, OldRefImageItemNode,
    OldTrainingSetListNode, OldTrainingSetItemNode, OldLayoutListNode, OldLayoutItemNode
} from './nodes/OldSchemaNodes';
import { DatasetConfigNode, DatasetListNode } from './nodes/NewSchemaNodes';
import { 
    ImageConfigNode, ImageListNode,
    TargetConfigNode, TargetListNode,
    ReferenceConfigNode, ReferenceListNode, ReferenceEntryNode,
    CaptionConfigNode, CaptionListNode,
    BatchConfigNode, BatchListNode
} from './nodes/ConfigSectionNodes';
import { parseGraph } from '../utils/graphParser';
import { DEFAULT_GLOBAL_CONFIG, SCRIPTS, TRANSLATIONS } from '../constants';
import { FlowContext, Language, Theme } from './FlowContext';
import { TrainingOutputModal } from './TrainingOutputModal';

// Logic for Auto-Connecting Nodes
const NODE_CONNECTION_MAP: Record<string, Record<string, { type: string; label: string }[]>> = {
  scriptNode: {
    dir_out: [{ type: 'directoryNode', label: 'Directory Setup' }],
    lora_out: [{ type: 'loraNode', label: 'LoRA Config' }],
    misc_out: [{ type: 'miscNode', label: 'Misc Settings' }],
    json_out: [{ type: 'datasetListNode', label: 'Dataset List' }],
    old_out: [{ type: 'oldSchemaRoot', label: 'Old Schema Root' }],
  },
  datasetListNode: {
    __default__: [{ type: 'datasetConfigNode', label: 'Dataset Config' }]
  },
  datasetConfigNode: {
    image_config: [{ type: 'imageListNode', label: 'Image List' }],
    target_config: [{ type: 'targetListNode', label: 'Target List' }],
    reference_config: [{ type: 'referenceListNode', label: 'Ref List' }],
    caption_config: [{ type: 'captionListNode', label: 'Caption List' }],
    batch_config: [{ type: 'batchListNode', label: 'Batch List' }]
  },
  imageListNode: { __default__: [{ type: 'imageConfigNode', label: 'Image Item' }] },
  targetListNode: { __default__: [{ type: 'targetConfigNode', label: 'Target Item' }] },
  referenceListNode: { __default__: [{ type: 'referenceConfigNode', label: 'Ref Group' }] },
  captionListNode: { __default__: [{ type: 'captionConfigNode', label: 'Caption Item' }] },
  batchListNode: { __default__: [{ type: 'batchConfigNode', label: 'Batch Item' }] },
  referenceConfigNode: { __default__: [{ type: 'referenceEntryNode', label: 'Ref Source' }] },
  
  oldSchemaRoot: {
    image_configs: [{ type: 'oldImageListNode', label: 'Image List (Old)' }],
    caption_configs: [{ type: 'oldCaptionListNode', label: 'Caption List (Old)' }],
    training_set: [{ type: 'oldTrainingSetListNode', label: 'Training Set List (Old)' }]
  },
  oldImageListNode: { __default__: [{ type: 'oldImageItemNode', label: 'Image Item (Old)' }] },
  oldCaptionListNode: { __default__: [{ type: 'oldCaptionItemNode', label: 'Caption Item (Old)' }] },
  oldCaptionItemNode: { ref_list: [{ type: 'oldRefImageListNode', label: 'Ref Image List (Old)' }] },
  oldRefImageListNode: { __default__: [{ type: 'oldRefImageItemNode', label: 'Ref Image Item (Old)' }] },
  oldTrainingSetListNode: { __default__: [{ type: 'oldTrainingSetItemNode', label: 'Training Set Item (Old)' }] },
  oldTrainingSetItemNode: { layout_list: [{ type: 'oldLayoutListNode', label: 'Layout List (Old)' }] },
  oldLayoutListNode: { __default__: [{ type: 'oldLayoutItemNode', label: 'Layout Item (Old)' }] }
};

const nodeTypes = {
  scriptNode: ScriptNode,
  directoryNode: DirectoryNode,
  loraNode: LoRANode,
  miscNode: MiscNode,
  configTemplateNode: ConfigTemplateNode,
  
  // New Schema Nodes
  datasetConfigNode: DatasetConfigNode,
  datasetListNode: DatasetListNode,
  imageConfigNode: ImageConfigNode, imageListNode: ImageListNode,
  targetConfigNode: TargetConfigNode, targetListNode: TargetListNode,
  referenceConfigNode: ReferenceConfigNode, referenceListNode: ReferenceListNode, referenceEntryNode: ReferenceEntryNode,
  captionConfigNode: CaptionConfigNode, captionListNode: CaptionListNode,
  batchConfigNode: BatchConfigNode, batchListNode: BatchListNode,

  // Old Schema Nodes
  oldSchemaRoot: OldSchemaRootNode,
  oldImageListNode: OldImageListNode, oldImageItemNode: OldImageItemNode,
  oldCaptionListNode: OldCaptionListNode, oldCaptionItemNode: OldCaptionItemNode, oldRefImageListNode: OldRefImageListNode, oldRefImageItemNode: OldRefImageItemNode,
  oldTrainingSetListNode: OldTrainingSetListNode, oldTrainingSetItemNode: OldTrainingSetItemNode, oldLayoutListNode: OldLayoutListNode, oldLayoutItemNode: OldLayoutItemNode
};

// Default empty initial nodes and edges - will be replaced by default workflow
const initialNodes: Node[] = [];

const initialEdges: Edge[] = [];

const defaultEdgeOptions = {
    type: 'default',
    animated: false,
    style: { strokeWidth: 2, stroke: '#a1a1aa' },
    interactionWidth: 15,
};



export const FlowEditor = () => {
  const [nodes, setNodes] = useState<Node[]>(initialNodes);
  const [edges, setEdges] = useState<Edge[]>(initialEdges);
  const [output, setOutput] = useState<{json: string, cmd: string}>({ json: '', cmd: '' });
  const [refreshCount, setRefreshCount] = useState(0);
  const [clipboard, setClipboard] = useState<Node[]>([]);
  
  // UI State
  const [showPalette, setShowPalette] = useState(false);
  const [showOutput, setShowOutput] = useState(true);
  const [showAbout, setShowAbout] = useState(false);
  const [showTrainingOutput, setShowTrainingOutput] = useState(false);
  const [showChat, setShowChat] = useState(false);
  const [showLoadWorkflow, setShowLoadWorkflow] = useState(false);
  const [lang, setLang] = useState<Language>('en');
  const [theme, setTheme] = useState<Theme>('dark');
  const [isSpacePressed, setIsSpacePressed] = useState(false);
  
  // History
  const [history, setHistory] = useState<{nodes: Node[], edges: Edge[]}[]>([]);
  const [future, setFuture] = useState<{nodes: Node[], edges: Edge[]}[]>([]);

  // Auto-connect / Menu State
  const [contextMenu, setContextMenu] = useState<{x: number, y: number, options: {type: string, label: string}[], sourceId: string, handleId: string} | null>(null);
  const connectingNodeId = useRef<string | null>(null);
  const connectingHandleId = useRef<string | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { project, setViewport, getViewport } = useReactFlow();

  // Helper for translations
  const t = (key: keyof typeof TRANSLATIONS.en) => TRANSLATIONS[lang][key] || key;

  // Theme Toggle Effect
  useEffect(() => {
      const root = document.documentElement;
      if (theme === 'dark') {
          root.classList.add('dark');
      } else {
          root.classList.remove('dark');
      }
  }, [theme]);

  const toggleTheme = () => setTheme(prev => prev === 'dark' ? 'light' : 'dark');
  const toggleLang = () => setLang(prev => prev === 'en' ? 'cn' : 'en');

  const snapshot = useCallback(() => {
    setHistory(prev => [...prev.slice(-20), { nodes, edges }]); 
    setFuture([]);
  }, [nodes, edges]);

  const onNodesChange: OnNodesChange = useCallback(
    (changes: NodeChange[]) => {
      // If Space is pressed, we ignore position changes for dragged nodes
      // enabling us to 'hijack' the drag to pan the viewport instead.
      if (isSpacePressed) {
          const filtered = changes.filter(c => c.type !== 'position');
          if (filtered.length < changes.length) {
              // Position change blocked.
          }
          if (filtered.length === 0) return;
          if (filtered.some(c => c.type === 'remove')) snapshot();
          setNodes((nds) => applyNodeChanges(filtered, nds));
      } else {
          if (changes.some(c => c.type === 'remove')) snapshot();
          setNodes((nds) => applyNodeChanges(changes, nds));
      }
    },
    [snapshot, isSpacePressed]
  );

  const onEdgesChange: OnEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      if (changes.some(c => c.type === 'remove')) snapshot();
      setEdges((eds) => applyEdgeChanges(changes, eds));
    },
    [snapshot]
  );

  const onNodeDragStart = useCallback(() => {
      if(!isSpacePressed) snapshot();
  }, [snapshot, isSpacePressed]);

  const onNodeDrag = useCallback((event: React.MouseEvent, node: Node) => {
      if (isSpacePressed) {
          const { x, y, zoom } = getViewport();
          // Pan viewport
          setViewport({
              x: x + event.movementX,
              y: y + event.movementY,
              zoom
          });
      }
  }, [isSpacePressed, getViewport, setViewport]);

  const updateNodeData = useCallback((id: string, key: string, value: any) => {
    console.log(`Updating node ${id}: setting ${key} to`, value);
    setNodes((nds) =>
      nds.map((node) => {
        if (node.id !== id) return node;
        // Create a completely new data object to ensure React detects the change
        const newData = { ...node.data };
        newData[key] = value;
        const newNode = { ...node, data: newData };
        console.log('Updated node:', newNode);
        return newNode;
      })
    );
  }, [setNodes]);

  const toggleLock = useCallback((id: string) => {
      setNodes((nds) =>
        nds.map((node) => {
            if (node.id !== id) return node;
            const newPinned = !node.data.pinned;
            return {
                ...node,
                // If not pinned, set draggable to undefined so it falls back to global ReactFlow 'nodesDraggable' setting
                draggable: !newPinned ? undefined : false,
                data: { ...node.data, pinned: newPinned }
            };
        })
      );
  }, [setNodes]);

  const onUndo = useCallback(() => {
    if (history.length === 0) return;
    const previous = history[history.length - 1];
    setFuture(prev => [{ nodes, edges }, ...prev]);
    setHistory(history.slice(0, -1));
    setNodes(previous.nodes);
    setEdges(previous.edges);
  }, [history, nodes, edges]);

  const onRedo = useCallback(() => {
    if (future.length === 0) return;
    const next = future[0];
    setFuture(future.slice(1));
    setHistory(prev => [...prev, { nodes, edges }]);
    setNodes(next.nodes);
    setEdges(next.edges);
  }, [future, nodes, edges]);

  // Load the default workflow on component mount
  useEffect(() => {
    const loadDefaultWorkflow = async () => {
      try {
        // Fetch the default workflow file
        const response = await fetch('./default_single_workflow.json');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const defaultWorkflow = await response.json();

        // Set the default workflow nodes and edges
        setNodes(defaultWorkflow.nodes);
        setEdges(defaultWorkflow.edges);

        // Add to history
        snapshot();
      } catch (error) {
        console.error('Failed to load default workflow:', error);

        // If loading fails, use the original initial nodes and edges as fallback
        setNodes([
          // Source (Left)
          { id: 'script', type: 'scriptNode', position: { x: 50, y: 300 }, data: { scriptName: SCRIPTS[2].name, configPath: "config_new.json" } },

          // First Layer (Target)
          { id: 'dir', type: 'directoryNode', position: { x: 450, y: 50 }, data: { 
              pretrained_model_name_or_path: DEFAULT_GLOBAL_CONFIG.pretrained_model_name_or_path,
              output_dir: DEFAULT_GLOBAL_CONFIG.output_dir,
              logging_dir: DEFAULT_GLOBAL_CONFIG.logging_dir,
              save_name: DEFAULT_GLOBAL_CONFIG.save_name,
              train_data_dir: DEFAULT_GLOBAL_CONFIG.train_data_dir,
              model_path: DEFAULT_GLOBAL_CONFIG.model_path,
              vae_path: DEFAULT_GLOBAL_CONFIG.vae_path
          } },
          { id: 'lora', type: 'loraNode', position: { x: 450, y: 350 }, data: { 
              rank: DEFAULT_GLOBAL_CONFIG.rank,
              rank_alpha: DEFAULT_GLOBAL_CONFIG.rank_alpha,
              use_lokr: DEFAULT_GLOBAL_CONFIG.use_lokr,
              lokr_factor: DEFAULT_GLOBAL_CONFIG.lokr_factor,
              lora_layers: DEFAULT_GLOBAL_CONFIG.lora_layers
          } },
          { id: 'misc', type: 'miscNode', position: { x: 450, y: 650 }, data: { 
              resolution: DEFAULT_GLOBAL_CONFIG.resolution,
              train_batch_size: DEFAULT_GLOBAL_CONFIG.train_batch_size,
              num_train_epochs: DEFAULT_GLOBAL_CONFIG.num_train_epochs,
              learning_rate: DEFAULT_GLOBAL_CONFIG.learning_rate,
              optimizer: DEFAULT_GLOBAL_CONFIG.optimizer,
              mixed_precision: DEFAULT_GLOBAL_CONFIG.mixed_precision,
              seed: DEFAULT_GLOBAL_CONFIG.seed,
              gradient_checkpointing: DEFAULT_GLOBAL_CONFIG.gradient_checkpointing,
              caption_dropout: DEFAULT_GLOBAL_CONFIG.caption_dropout,
              save_model_epochs: DEFAULT_GLOBAL_CONFIG.save_model_epochs,
              blocks_to_swap: DEFAULT_GLOBAL_CONFIG.blocks_to_swap,
              recreate_cache: DEFAULT_GLOBAL_CONFIG.recreate_cache,
              validation_epochs: DEFAULT_GLOBAL_CONFIG.validation_epochs,
              validation_ratio: DEFAULT_GLOBAL_CONFIG.validation_ratio,
              weighting_scheme: DEFAULT_GLOBAL_CONFIG.weighting_scheme,
              logit_mean: DEFAULT_GLOBAL_CONFIG.logit_mean,
              logit_std: DEFAULT_GLOBAL_CONFIG.logit_std,
              mode_scale: DEFAULT_GLOBAL_CONFIG.mode_scale,
              guidance_scale: DEFAULT_GLOBAL_CONFIG.guidance_scale,

              mask_dropout: DEFAULT_GLOBAL_CONFIG.mask_dropout,
              freeze_transformer_layers: DEFAULT_GLOBAL_CONFIG.freeze_transformer_layers,
              freeze_single_transformer_layers: DEFAULT_GLOBAL_CONFIG.freeze_single_transformer_layers
          } },
          { id: 'ds_list', type: 'datasetListNode', position: { x: 450, y: 1000 }, data: { slots: ['dataset_0'] } },

          // Second Layer (Target of List)
          { id: 'dataset', type: 'datasetConfigNode', position: { x: 800, y: 1000 }, data: { train_data_dir: "F:/ImageSet/longcat" } },

          // Third Layer (Component Lists)
          { id: 'img_list', type: 'imageListNode', position: { x: 1200, y: 800 }, data: { slots: ['img_0'] } },
          { id: 'tgt_list', type: 'targetListNode', position: { x: 1200, y: 1000 }, data: { slots: ['tgt_0'] } },
          { id: 'ref_list', type: 'referenceListNode', position: { x: 1200, y: 1200 }, data: { slots: ['ref_0'] } },
          { id: 'cap_list', type: 'captionListNode', position: { x: 1200, y: 1500 }, data: { slots: ['cap_0'] } },
          { id: 'batch_list', type: 'batchListNode', position: { x: 1200, y: 1800 }, data: { slots: ['batch_0'] } },

          // Fourth Layer (Items)
          { id: 'img_1', type: 'imageConfigNode', position: { x: 1600, y: 800 }, data: { key: 'train', suffix: '_T' } },
          { id: 'tgt_1', type: 'targetConfigNode', position: { x: 1600, y: 1000 }, data: { key: 'train', image: 'train', from_image: '' } },

          // Reference Split: Group + Entry
          { id: 'ref_group_1', type: 'referenceConfigNode', position: { x: 1600, y: 1200 }, data: { key: 'train_ref', slots: ['entry_0'] } },
          { id: 'ref_entry_1', type: 'referenceEntryNode', position: { x: 1950, y: 1200 }, data: { sample_type: 'from_same_name', image: 'train' } },

          { id: 'cap_1', type: 'captionConfigNode', position: { x: 1600, y: 1500 }, data: { key: 'train', ext: '.txt', image: 'train' } },
          { id: 'batch_1', type: 'batchConfigNode', position: { x: 1600, y: 1800 }, data: { target_config: 'train', caption_config: 'train', caption_dropout: 0.1 } },
        ]);

        setEdges([
          // Script (Source Right) -> Global (Target Left)
          { id: 'e1', source: 'script', sourceHandle: 'dir_out', target: 'dir', targetHandle: 'in' },
          { id: 'e2', source: 'script', sourceHandle: 'lora_out', target: 'lora', targetHandle: 'in' },
          { id: 'e3', source: 'script', sourceHandle: 'misc_out', target: 'misc', targetHandle: 'in' },
          { id: 'e4', source: 'script', sourceHandle: 'json_out', target: 'ds_list', targetHandle: 'in' },

          // Dataset List (Source Right) -> Dataset (Target Left)
          { id: 'e5', source: 'ds_list', sourceHandle: 'dataset_0', target: 'dataset', targetHandle: 'in' },

          // Dataset (Source Right) -> Component Lists (Target Left)
          { id: 'e6', source: 'dataset', sourceHandle: 'image_config', target: 'img_list', targetHandle: 'in' },
          { id: 'e7', source: 'dataset', sourceHandle: 'target_config', target: 'tgt_list', targetHandle: 'in' },
          { id: 'e8', source: 'dataset', sourceHandle: 'reference_config', target: 'ref_list', targetHandle: 'in' },
          { id: 'e9', source: 'dataset', sourceHandle: 'caption_config', target: 'cap_list', targetHandle: 'in' },
          { id: 'e10', source: 'dataset', sourceHandle: 'batch_config', target: 'batch_list', targetHandle: 'in' },

          // Component List (Source Right) -> Item (Target Left)
          { id: 'e11', source: 'img_list', sourceHandle: 'img_0', target: 'img_1', targetHandle: 'in' },
          { id: 'e12', source: 'tgt_list', sourceHandle: 'tgt_0', target: 'tgt_1', targetHandle: 'in' },

          // Reference: List -> Group -> Entry
          { id: 'e13', source: 'ref_list', sourceHandle: 'ref_0', target: 'ref_group_1', targetHandle: 'in' },
          { id: 'e13b', source: 'ref_group_1', sourceHandle: 'entry_0', target: 'ref_entry_1', targetHandle: 'in' },

          { id: 'e14', source: 'cap_list', sourceHandle: 'cap_0', target: 'cap_1', targetHandle: 'in' },
          { id: 'e15', source: 'batch_list', sourceHandle: 'batch_0', target: 'batch_1', targetHandle: 'in' },
        ]);

        snapshot();
      }
    };

    loadDefaultWorkflow();
  }, []); // Empty dependency array means this will run once on component mount

  const saveWorkflow = () => {
      const data = { nodes, edges, timestamp: Date.now(), version: "1.0" };
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `workflow_${new Date().toISOString().slice(0,10)}.json`;
      link.click();
      URL.revokeObjectURL(url);
  };

  const saveWorkflowAsTemplate = async (name: string, workflow: { nodes: any[]; edges: any[] }) => {
      try {
        // Save to backend as custom template
        const data = { ...workflow, timestamp: Date.now(), version: "1.0" };
        
        const response = await fetch('/api/templates/custom', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            name: name,
            workflow: data
          })
        });
        
        if (!response.ok) {
          throw new Error('Failed to save template');
        }
        
        const result = await response.json();
        console.log('Template saved:', result);
        
        // Refresh templates in LoadWorkflowPanel
        window.dispatchEvent(new CustomEvent('refreshTemplates'));
        
        return Promise.resolve();
      } catch (error) {
        console.error('Error saving template:', error);
        return Promise.reject(error);
      }
  };

  const deleteTemplate = async (path: string) => {
      try {
        // Extract template name from path
        const templateName = path.split('/').pop();
        
        if (!templateName) {
          throw new Error('Invalid template path');
        }
        
        // Delete template via backend API
        const response = await fetch(`/api/templates/custom/${templateName}`, {
          method: 'DELETE'
        });
        
        if (!response.ok) {
          throw new Error('Failed to delete template');
        }
        
        const result = await response.json();
        console.log('Template deleted:', result);
        
        return Promise.resolve();
      } catch (error) {
        console.error('Error deleting template:', error);
        return Promise.reject(error);
      }
  };

  const loadWorkflowFile = (file: File) => {
      const reader = new FileReader();
      reader.onload = (e) => {
          try {
              const content = e.target?.result as string;
              const data = JSON.parse(content);
              if (data.nodes && data.edges) {
                  snapshot();
                  setNodes(data.nodes);
                  setEdges(data.edges);
              } else {
                  alert("Invalid workflow file format.");
              }
          } catch (err) {
              alert("Failed to parse JSON file.");
          }
      };
      reader.readAsText(file);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files && e.target.files[0]) loadWorkflowFile(e.target.files[0]);
      if(fileInputRef.current) fileInputRef.current.value = '';
  };
  
  const runTraining = async () => {
      const scriptNode = nodes.find(n => n.type === 'scriptNode');
      const scriptName = scriptNode?.data.scriptName;
      const configPath = scriptNode?.data.configPath || "config.json";

      if (!scriptName) {
          alert("No script selected! Please ensure a Script Loader node exists.");
          return;
      }

      // Auto-save the current workflow before training
      try {
          await saveWorkflowAsTemplate('autosave_temp.json', { nodes, edges });
          console.log('Workflow auto-saved before training');
      } catch (saveError) {
          console.error('Failed to auto-save workflow:', saveError);
          // Continue with training even if auto-save fails
      }

      // Parse the configuration from the output JSON generated by the graph parser
      try {
          const configData = JSON.parse(output.json);
          // The modal will handle the WebSocket connection and training start
          setShowTrainingOutput(true);
      } catch (e) {
          console.error(e);
          alert("Failed to parse configuration. Please ensure your graph is properly connected.");
      }
  };

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback((event: React.DragEvent) => {
      event.preventDefault();
      if (event.dataTransfer.files && event.dataTransfer.files.length > 0) {
           loadWorkflowFile(event.dataTransfer.files[0]);
      }
    }, [snapshot]
  );

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
        const isCtrl = e.ctrlKey || e.metaKey;
        const target = e.target as HTMLElement;
        const isInput = target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT';

        if (e.code === 'Space' && !e.repeat && !isInput) {
             e.preventDefault();
             setIsSpacePressed(true);
        }

        if (isCtrl && e.key.toLowerCase() === 'c' && !isInput) {
            const selected = nodes.filter(n => n.selected);
            if (selected.length > 0) setClipboard(selected);
        }

        if (isCtrl && e.key.toLowerCase() === 'v' && !isInput && clipboard.length > 0) {
            snapshot();
            setNodes(nds => nds.map(n => ({...n, selected: false})));
            const newNodes = clipboard.map(node => ({
                ...node,
                id: `${node.type}_${Math.random().toString(36).substr(2, 6)}`,
                position: { x: node.position.x + 50, y: node.position.y + 50 },
                selected: true,
                data: { ...node.data }
            }));
            setNodes(nds => [...nds, ...newNodes]);
        }

        if (isCtrl && e.key.toLowerCase() === 'z' && !isInput) {
            e.preventDefault();
            e.shiftKey ? onRedo() : onUndo();
        }
        if (isCtrl && e.key.toLowerCase() === 'y' && !isInput) {
            e.preventDefault();
            onRedo();
        }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
        if (e.code === 'Space') {
            setIsSpacePressed(false);
        }
    };

    const handleBlur = () => setIsSpacePressed(false);

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    window.addEventListener('blur', handleBlur);
    return () => {
        window.removeEventListener('keydown', handleKeyDown);
        window.removeEventListener('keyup', handleKeyUp);
        window.removeEventListener('blur', handleBlur);
    };
  }, [nodes, clipboard, snapshot, onUndo, onRedo]);

  const deleteNode = useCallback((id: string) => {
      snapshot();
      setNodes(nds => nds.filter(n => n.id !== id));
      setEdges(eds => eds.filter(e => e.source !== id && e.target !== id));
  }, [snapshot, setNodes, setEdges]);

  const copyNode = useCallback((id: string) => {
      snapshot();
      const node = nodes.find(n => n.id === id);
      if(!node) return;
      const newNode = {
          ...node,
          id: `${node.type}_${Math.random().toString(36).substr(2,6)}`,
          position: { x: node.position.x + 50, y: node.position.y + 50 },
          selected: true
      };
      setNodes(nds => [...nds.map(n => ({...n, selected: false})), newNode]);
  }, [nodes, snapshot, setNodes]);

  const onConnect = useCallback((params: Connection) => {
      console.log('Connection made:', params);
      snapshot();
      setEdges((eds) => addEdge(params, eds.filter(e => !(e.target === params.target && e.targetHandle === params.targetHandle))));
      console.log('Edges after connection:', edges);
  }, [setEdges, snapshot, edges]);

  // Create a deep hash of nodes and edges to detect actual data changes
  const getNodeDataHash = useCallback(() => {
    // Create a simplified representation of the graph for comparison
    const nodeData = nodes.map(n => ({
      id: n.id,
      type: n.type,
      data: n.data
    })).sort((a, b) => a.id.localeCompare(b.id));
    
    const edgeData = edges.map(e => ({
      id: e.id,
      source: e.source,
      target: e.target,
      sourceHandle: e.sourceHandle,
      targetHandle: e.targetHandle
    })).sort((a, b) => a.id.localeCompare(b.id));
    
    return JSON.stringify({ nodes: nodeData, edges: edgeData });
  }, [nodes, edges]);

  const [lastHash, setLastHash] = useState('');

  useEffect(() => {
    const currentHash = getNodeDataHash();
    if (currentHash !== lastHash) {
      console.log('Nodes or edges data changed, reparsing graph');
      setLastHash(currentHash);
      const result = parseGraph(nodes, edges);
      setOutput({ json: JSON.stringify(result.config, null, 2), cmd: result.command });
    }
  }, [nodes, edges, getNodeDataHash, lastHash]);

  // Manual refresh function for the refresh button
  const refreshOutput = useCallback(() => {
    // Use a small delay to ensure any pending state updates are processed
    setTimeout(() => {
      console.log('Refreshing output with nodes:', nodes);
      console.log('Refreshing output with edges:', edges);
      
      // Check if we have the latest data by looking at a specific node
      const scriptNode = nodes.find(n => n.type === 'scriptNode');
      if (scriptNode) {
        console.log('Script node data:', scriptNode.data);
      }
      
      // Force a complete reparse with a deep copy
      try {
        const nodesCopy = JSON.parse(JSON.stringify(nodes));
        const edgesCopy = JSON.parse(JSON.stringify(edges));
        const result = parseGraph(nodesCopy, edgesCopy);
        console.log('Parsed result:', result);
        setOutput({ json: JSON.stringify(result.config, null, 2), cmd: result.command });
        setRefreshCount(prev => prev + 1);
        // Also update the hash to prevent automatic refresh from triggering
        const currentHash = JSON.stringify({
          nodes: nodesCopy.map(n => ({ id: n.id, type: n.type, data: n.data })).sort((a, b) => a.id.localeCompare(b.id)),
          edges: edgesCopy.map(e => ({ id: e.id, source: e.source, target: e.target, sourceHandle: e.sourceHandle, targetHandle: e.targetHandle })).sort((a, b) => a.id.localeCompare(b.id))
        });
        setLastHash(currentHash);
      } catch (error) {
        console.error('Error during refresh:', error);
      }
    }, 10);
  }, [nodes, edges]);

  // Extended addNode to support auto-wiring
  const addNode = (type: string, label: string, position?: {x: number, y: number}, sourceInfo?: { nodeId: string, handleId: string }) => {
      snapshot();
      const id = `${type}_${Math.random().toString(36).substr(2, 6)}`;
      const pos = position || { x: Math.random() * 400 + 100, y: Math.random() * 400 + 100 };
      
      setNodes((nds) => nds.concat({
        id, type, position: pos, data: { label },
      }));

      if(sourceInfo) {
          setEdges((eds) => addEdge({
              source: sourceInfo.nodeId,
              sourceHandle: sourceInfo.handleId,
              target: id,
              targetHandle: 'in', // Assume standard input is 'in'
          }, eds));
      }
  };

  // --- Auto-Connect Logic ---
  const onConnectStart = useCallback((_: any, params: OnConnectStartParams) => {
      connectingNodeId.current = params.nodeId;
      connectingHandleId.current = params.handleId;
  }, []);

  const onConnectEnd = useCallback((event: any) => {
    const targetIsPane = event.target.classList.contains('react-flow__pane');
    
    if (targetIsPane && connectingNodeId.current && connectingHandleId.current) {
        const srcNode = nodes.find(n => n.id === connectingNodeId.current);
        if(!srcNode) return;

        const mapping = NODE_CONNECTION_MAP[srcNode.type || ''];
        if(mapping) {
            // Find specific handle mapping or default
            const options = mapping[connectingHandleId.current] || mapping['__default__'];
            
            if(options && options.length > 0) {
                // Calculate position
                const { top, left } = event.target.getBoundingClientRect();
                const position = project({ 
                    x: event.clientX - left, 
                    y: event.clientY - top 
                });

                if(options.length === 1) {
                    // Auto Create
                    addNode(options[0].type, options[0].label, position, { 
                        nodeId: connectingNodeId.current, 
                        handleId: connectingHandleId.current 
                    });
                } else {
                    // Show Menu
                    setContextMenu({
                        x: event.clientX,
                        y: event.clientY,
                        options,
                        sourceId: connectingNodeId.current,
                        handleId: connectingHandleId.current
                    });
                }
            }
        }
    }
    // Reset
    connectingNodeId.current = null;
    connectingHandleId.current = null;
  }, [nodes, project]);

  const onPaneClick = useCallback(() => setContextMenu(null), []);

  const handleMenuSelect = (option: {type: string, label: string}) => {
      if(contextMenu) {
        // We need to re-calculate project here since contextMenu uses client coords
        const reactFlowPane = document.querySelector('.react-flow__pane');
        if(reactFlowPane) {
             const { top, left } = reactFlowPane.getBoundingClientRect();
             const position = project({ 
                x: contextMenu.x - left, 
                y: contextMenu.y - top 
            });
            addNode(option.type, option.label, position, { 
                nodeId: contextMenu.sourceId, 
                handleId: contextMenu.handleId 
            });
        }
      }
      setContextMenu(null);
  };

  // Helper for Button UI
  const IconButton = ({ onClick, active, title, icon: Icon, color = 'blue' }: any) => (
      <button 
        onClick={onClick}
        title={title}
        className={`p-2 rounded-lg transition-all duration-200 group
            ${active 
                ? `bg-${color}-500/20 text-${color}-500 ring-1 ring-${color}-500/50` 
                : 'text-zinc-500 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-white hover:bg-black/5 dark:hover:bg-white/10'
            }`}
      >
          <Icon size={18} />
      </button>
  );

  return (
    <FlowContext.Provider value={{
        updateNodeData, deleteNode, copyNode, toggleLock, undo: onUndo, redo: onRedo,
        canUndo: history.length > 0, canRedo: future.length > 0,
        lang, theme
    }}>
        <div className="relative h-screen w-screen bg-zinc-200 dark:bg-zinc-950 overflow-hidden transition-colors duration-300" onDrop={onDrop} onDragOver={onDragOver}>
            
            {/* Minimalist Left Sidebar */}
            <div className="absolute top-4 left-4 z-50 flex flex-col items-center gap-1.5 py-2 px-1 rounded-xl shadow-lg border border-zinc-300 dark:border-zinc-800 bg-white/30 dark:bg-black/20 backdrop-blur-md w-12 transition-all">
                <IconButton onClick={() => { setShowOutput(!showOutput); setShowPalette(false); setShowAbout(false); setShowLoadWorkflow(false); }} active={showOutput} title={t('output')} icon={Terminal} color="orange" />
                <IconButton onClick={() => { setShowPalette(!showPalette); setShowOutput(false); setShowAbout(false); setShowLoadWorkflow(false); }} active={showPalette} title={t('palette')} icon={LayoutGrid} color="blue" />
                
                <div className="w-6 h-px bg-zinc-300 dark:bg-zinc-700 my-1"></div>
                
                <IconButton onClick={() => { setShowLoadWorkflow(!showLoadWorkflow); setShowOutput(false); setShowPalette(false); setShowAbout(false); }} active={showLoadWorkflow} title={t('load')} icon={FolderOpen} />

                <div className="w-6 h-px bg-zinc-300 dark:bg-zinc-700 my-1"></div>

                <IconButton onClick={() => { setShowAbout(!showAbout); setShowPalette(false); setShowOutput(false); setShowLoadWorkflow(false); }} active={showAbout} title={t('about')} icon={Info} color="purple" />

                {/* <div className="w-6 h-px bg-zinc-300 dark:bg-zinc-700 my-1"></div>
                
                <IconButton onClick={() => { setShowChat(!showChat); setShowPalette(false); setShowOutput(false); setShowAbout(false); }} active={showChat} title="Chat Test" icon={Terminal} color="cyan" /> */}

                <div className="w-6 h-px bg-zinc-300 dark:bg-zinc-700 my-1"></div>
                
                <IconButton onClick={toggleTheme} title={theme === 'dark' ? "Light Mode" : "Dark Mode"} icon={theme === 'dark' ? Moon : Sun} />
                <button 
                    onClick={toggleLang}
                    title="Switch Language"
                    className="p-2 rounded-lg text-xs font-bold text-zinc-500 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-white hover:bg-black/5 dark:hover:bg-white/10 transition-colors"
                >
                    {lang === 'en' ? 'EN' : '中'}
                </button>
            </div>

            {contextMenu && (
                <div style={{ top: contextMenu.y, left: contextMenu.x }} className="absolute z-[60] bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-lg shadow-xl p-1 min-w-[150px] flex flex-col gap-0.5 animate-in fade-in zoom-in-95 duration-100 origin-top-left">
                    <div className="px-2 py-1 text-[10px] uppercase font-bold text-zinc-400 dark:text-zinc-500 border-b border-zinc-100 dark:border-zinc-700/50 mb-1">Create Node</div>
                    {contextMenu.options.map(opt => (
                        <button 
                            key={opt.type} 
                            onClick={() => handleMenuSelect(opt)} 
                            className="text-left px-2 py-1.5 text-xs text-zinc-700 dark:text-zinc-200 hover:bg-blue-50 dark:hover:bg-blue-900/30 hover:text-blue-600 dark:hover:text-blue-400 rounded transition-colors"
                        >
                            {opt.label}
                        </button>
                    ))}
                </div>
            )}

            {/* Main Canvas */}
            <ReactFlow
                nodes={nodes}
                edges={edges}
                nodeTypes={nodeTypes}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onNodeDragStart={onNodeDragStart}
                onNodeDrag={onNodeDrag}
                onConnect={onConnect}
                onConnectStart={onConnectStart}
                onConnectEnd={onConnectEnd}
                onPaneClick={onPaneClick}
                deleteKeyCode={['Backspace', 'Delete']}
                selectionKeyCode="Shift"
                selectionMode={SelectionMode.Partial}
                multiSelectionKeyCode={['Control', 'Meta', 'Shift']}
                nodesDraggable={!isSpacePressed}
                panOnDrag={true}
                defaultEdgeOptions={defaultEdgeOptions}
                fitView
                minZoom={0.1}
                connectionRadius={40}
                proOptions={{ hideAttribution: true }}
                className="absolute inset-0"
            >
                <Background color={theme === 'dark' ? "#27272a" : "#d4d4d8"} gap={20} />
            </ReactFlow>

            {/* About Panel */}
            {showAbout && (
                <AboutPanel
                    isOpen={showAbout}
                    onClose={() => setShowAbout(false)}
                    lang={lang}
                />
            )}

            {/* Training Output Modal */}
            {showTrainingOutput && (
              <TrainingOutputModal
                isOpen={showTrainingOutput}
                onClose={() => setShowTrainingOutput(false)}
                config={(() => {
                  try {
                    return JSON.parse(output.json || '{}');
                  } catch {
                    return {};
                  }
                })()}
                scriptName={nodes.find(n => n.type === 'scriptNode')?.data.scriptName || ''}
                configPath={nodes.find(n => n.type === 'scriptNode')?.data.configPath || 'config.json'}
              />
            )}



            {/* Chat Panel */}
            {showChat && <ChatPanel onClose={() => setShowChat(false)} />}

            {/* Floating Palette Panel */}
            {showPalette && (
                <div className="absolute left-20 top-4 bottom-4 w-80 bg-white/30 dark:bg-zinc-900/40 backdrop-blur-sm border border-zinc-200 dark:border-zinc-800 rounded-2xl shadow-2xl flex flex-col z-40 animate-in slide-in-from-left-4 fade-in duration-200">
                    <div className="p-4 border-b border-zinc-200 dark:border-zinc-800 flex justify-between items-center bg-white/70 dark:bg-zinc-900/70 rounded-t-2xl">
                        <h2 className="text-zinc-600 dark:text-zinc-400 text-xs font-bold uppercase tracking-wider flex items-center gap-2">
                            <LayoutGrid size={14}/> {t('palette')}
                        </h2>
                        <button onClick={() => setShowPalette(false)} className="text-zinc-400 hover:text-zinc-900 dark:hover:text-white"><X size={14}/></button>
                    </div>
                    
                    <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-3 scrollbar-thin">
                        <div className="flex gap-2">
                            <button onClick={onUndo} disabled={history.length===0} className="flex-1 bg-white/80 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-300 p-2 rounded-lg text-xs font-medium hover:bg-white dark:hover:bg-zinc-700 disabled:opacity-50 border border-zinc-200 dark:border-zinc-700 transition-all active:scale-95 shadow-sm">{t('undo')}</button>
                            <button onClick={onRedo} disabled={future.length===0} className="flex-1 bg-white/80 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-300 p-2 rounded-lg text-xs font-medium hover:bg-white dark:hover:bg-zinc-700 disabled:opacity-50 border border-zinc-200 dark:border-zinc-700 transition-all active:scale-95 shadow-sm">{t('redo')}</button>
                        </div>

                        {/* Helper to render palette buttons */}
                        {[
                            { title: t('core'), items: [
                                { id: 'directoryNode', label: t('dir'), color: 'blue' },
                                { id: 'loraNode', label: t('lora'), color: 'purple' },
                                { id: 'miscNode', label: t('misc'), color: 'slate' }
                            ]},
                            { title: t('old_schema'), items: [
                                { id: 'oldSchemaRoot', label: t('old_root'), color: 'amber' },
                                { id: 'oldImageListNode', label: t('old_image_list'), color: 'amber' },
                                { id: 'oldImageItemNode', label: t('old_image_item'), color: 'amber' },
                                { id: 'oldCaptionListNode', label: t('old_caption_list'), color: 'amber' },
                                { id: 'oldCaptionItemNode', label: t('old_caption_item'), color: 'amber' },
                                { id: 'oldRefImageListNode', label: t('old_ref_list'), color: 'orange' },
                                { id: 'oldRefImageItemNode', label: t('old_ref_item'), color: 'orange' },
                                { id: 'oldTrainingSetListNode', label: t('old_train_list'), color: 'indigo' },
                                { id: 'oldTrainingSetItemNode', label: t('old_train_item'), color: 'indigo' },
                                { id: 'oldLayoutListNode', label: t('old_layout_list'), color: 'teal' },
                                { id: 'oldLayoutItemNode', label: t('old_layout_item'), color: 'teal' },
                            ], grid: true },
                            { title: t('structure'), items: [
                                { id: 'datasetListNode', label: t('ds_list'), color: 'cyan' },
                                { id: 'datasetConfigNode', label: t('ds_conf'), color: 'cyan' }
                            ]},
                            { title: t('components'), items: [
                                { id: 'imageListNode', label: t('img_list'), color: 'emerald' },
                                { id: 'imageConfigNode', label: t('img_item'), color: 'emerald' },
                                { id: 'targetListNode', label: t('tgt_list'), color: 'teal' },
                                { id: 'targetConfigNode', label: t('tgt_item'), color: 'teal' },
                                { id: 'referenceListNode', label: t('ref_list'), color: 'orange' },
                                { id: 'referenceConfigNode', label: t('ref_item'), color: 'orange' },
                                { id: 'referenceEntryNode', label: t('ref_src'), color: 'orange' },
                                { id: 'captionListNode', label: t('cap_list'), color: 'pink' },
                                { id: 'captionConfigNode', label: t('cap_item'), color: 'pink' },
                                { id: 'batchListNode', label: t('batch_list'), color: 'indigo' },
                                { id: 'batchConfigNode', label: t('batch_item'), color: 'indigo' },
                            ], grid: true }
                        ].map((section, idx) => (
                            <div key={idx}>
                                <h3 className="text-zinc-500 dark:text-zinc-300 text-[10px] uppercase font-bold mb-2 px-1">{section.title}</h3>
                                <div className={section.grid ? "grid grid-cols-2 gap-2" : "space-y-2"}>
                                    {section.items.map(item => (
                                        <button 
                                            key={item.id}
                                            onClick={() => addNode(item.id, item.label)} 
                                            className={`w-full bg-white dark:bg-zinc-800 text-${item.color}-700 dark:text-${item.color}-200 text-xs p-2.5 rounded-lg border border-zinc-200 dark:border-zinc-700 text-left hover:border-${item.color}-400 dark:hover:border-${item.color}-500 hover:shadow-md transition-all`}
                                        >
                                            {item.label}
                                        </button>
                                    ))}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Floating Output Panel */}
            {showOutput && (
                <div className="absolute left-20 top-4 bottom-4 w-[500px] bg-white/20 dark:bg-zinc-900/50 backdrop-blur-sm border border-zinc-200 dark:border-zinc-800 rounded-2xl shadow-2xl flex flex-col z-40 animate-in slide-in-from-left-4 fade-in duration-200">
                    <div className="p-4 border-b border-zinc-200 dark:border-zinc-800 flex justify-between items-center bg-white/70 dark:bg-zinc-900/70 rounded-t-2xl">
                        <h2 className="text-zinc-600 dark:text-zinc-400 text-xs font-bold uppercase tracking-wider flex items-center gap-2">
                            <Terminal size={14}/> {t('output')}
                        </h2>
                        <button onClick={() => setShowOutput(false)} className="text-zinc-400 hover:text-zinc-900 dark:hover:text-white"><X size={14}/></button>
                    </div>
                    
                    <div className="flex-1 overflow-hidden p-5 flex flex-col gap-6">
                        <div className="flex-shrink-0">
                            <div className="flex justify-between items-center mb-2">
                                <label className="text-[10px] uppercase text-blue-500 font-bold tracking-wider">{t('cmd')}</label>
                                <div className="flex gap-1">
                                    <button onClick={runTraining} className="text-[10px] bg-emerald-500 dark:bg-emerald-600 px-2 py-1 rounded text-white hover:bg-emerald-600 dark:hover:bg-emerald-700 transition-colors">{t('open_training_panel')}</button>
                                    <button onClick={() => navigator.clipboard.writeText(output.cmd)} className="text-[10px] bg-zinc-200 dark:bg-zinc-800 px-2 py-1 rounded text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-white transition-colors">{t('copy')}</button>
                                </div>
                            </div>
                            <div className="bg-white/90 dark:bg-black/80 p-4 rounded-xl border border-zinc-200 dark:border-zinc-700/50 text-green-600 dark:text-green-400 font-mono text-xs break-all selection:bg-green-200 dark:selection:bg-green-900 selection:text-black dark:selection:text-white leading-relaxed shadow-sm">
                                {output.cmd}
                            </div>
                        </div>

                        <div className="flex-1 flex flex-col min-h-0">
                            <div className="flex justify-between items-center mb-2 flex-shrink-0">
                                <label className="text-[10px] uppercase text-orange-500 font-bold tracking-wider">{t('json')}</label>
                                <div className="flex gap-1">
                                    <button onClick={() => navigator.clipboard.writeText(output.json)} className="text-[10px] bg-zinc-200 dark:bg-zinc-800 px-2 py-1 rounded text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-white transition-colors">{t('copy')}</button>
                                </div>
                            </div>
                            <pre className="flex-1 overflow-auto bg-white/90 dark:bg-black/80 p-4 rounded-xl border border-zinc-200 dark:border-zinc-700/50 text-zinc-600 dark:text-zinc-300 font-mono text-[10px] scrollbar-thin selection:bg-orange-200 dark:selection:bg-orange-900 selection:text-black dark:selection:text-white shadow-sm">
                                {output.json}
                            </pre>
                        </div>
                    </div>
                </div>
            )}

            {/* Load Workflow Panel */}
            {showLoadWorkflow && (
                <LoadWorkflowPanel
                    isOpen={showLoadWorkflow}
                    onClose={() => setShowLoadWorkflow(false)}
                    onLoadWorkflow={(workflow) => {
                        snapshot();
                        setNodes(workflow.nodes);
                        setEdges(workflow.edges);
                    }}
                    onSaveWorkflow={saveWorkflowAsTemplate}
                    onDeleteTemplate={deleteTemplate}
                    currentWorkflow={{ nodes, edges }}
                    setNodes={setNodes}
                    setEdges={setEdges}
                    lang={lang}
                />
            )}
        </div>
    </FlowContext.Provider>
  );
};