import { Node, Edge } from 'reactflow';

export enum ScriptType {
  OLD = 'old', // e.g. train_qwen_image.py
  NEW = 'new'  // e.g. train_longcat.py
}

// Data stored inside the nodes
export interface NodeData {
  label: string;
  value?: any;
  onChange?: (key: string, value: any) => void;
  [key: string]: any;
}

export interface TrainingConfig {
  script: string;
  config_path: string;
  // Global params
  pretrained_model_name_or_path?: string;
  output_dir?: string;
  save_name?: string;
  seed?: number;
  mixed_precision?: string;
  gradient_checkpointing?: boolean;
  optimizer?: string;
  learning_rate?: number;
  rank?: number;
  [key: string]: any;
  
  // Dynamic JSON parts (Old Schema or inside Dataset)
  image_configs?: Record<string, any>;
  caption_configs?: Record<string, any>;
  target_configs?: Record<string, any>;
  reference_configs?: Record<string, any>;
  batch_configs?: any[];
  
  // New Schema
  dataset_configs?: any[];
}
