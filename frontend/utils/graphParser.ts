import { Node, Edge } from 'reactflow';
import { TrainingConfig, ScriptType } from '../types';
import { SCRIPTS, DEFAULT_GLOBAL_CONFIG } from '../constants';

export const parseGraph = (nodes: Node[], edges: Edge[]): { config: TrainingConfig, command: string } => {
  console.log('Parsing graph with nodes:', nodes);
  console.log('Parsing graph with edges:', edges);
  
  // Log all edges from script node
  const scriptNode = nodes.find(n => n.type === 'scriptNode');
  if (scriptNode) {
    console.log('Edges from script node:', edges.filter(e => e.source === scriptNode.id));
  }
  
  // Log available node types
  const nodeTypes = nodes.map(n => n.type);
  console.log('Available node types:', [...new Set(nodeTypes)]);
  
  if (!scriptNode) {
    console.log('No script node found');
    return { config: {} as any, command: "Error: No Script Node found." };
  }
  
  console.log('Found script node:', scriptNode);

  const scriptName = scriptNode.data.scriptName || SCRIPTS[0].name;
  const configPath = scriptNode.data.configPath || "config.json";
  const scriptType = SCRIPTS.find(s => s.name === scriptName)?.type || ScriptType.OLD;

  let config: TrainingConfig = {
    script: scriptName,
    config_path: configPath,
    ...DEFAULT_GLOBAL_CONFIG
  };

  // Helper: Find target node connected to a specific Source Handle on Script Node
  const mergeFromSourceHandle = (handleId: string) => {
      console.log(`Looking for edge from script node ${scriptNode.id} with handle ${handleId}`);
      // Find edge where Source is ScriptNode and SourceHandle is handleId
      const edge = edges.find(e => e.source === scriptNode.id && e.sourceHandle === handleId);
      console.log(`Found edge for handle ${handleId}:`, edge);
      if (edge) {
          const node = nodes.find(n => n.id === edge.target);
          console.log(`Found target node for edge:`, node);
          if (node) {
              if (node.type === 'oldSchemaRoot') {
                  // Do not merge root data directly into config, root is a router.
                  console.log('Skipping oldSchemaRoot node');
              } else {
                  const { onChange, label, slots, ...data } = node.data;
                  console.log(`Merging data from node ${node.id}:`, data);
                  
                  // Selective merging based on node type to avoid conflicts
                  if (node.type === 'directoryNode') {
                      // Only merge directory-specific fields to avoid overwriting with defaults
                      const dirFields = [
                          'pretrained_model_name_or_path',
                          'output_dir',
                          'logging_dir',
                          'save_name',
                          'train_data_dir',
                          'model_path',
                          'vae_path'
                      ];
                      const dirData = {};
                      dirFields.forEach(field => {
                          if (data[field] !== undefined) {
                              dirData[field] = data[field];
                          }
                      });
                      config = { ...config, ...dirData };
                  } else if (node.type === 'loraNode') {
                      // Only merge LoRA-specific fields (excluding pretrained_model_name_or_path to avoid conflicts)
                      const loraFields = [
                          'rank',
                          'rank_alpha',
                          'use_lokr',
                          'lokr_factor',
                          'lora_layers'
                      ];
                      const loraData = {};
                      loraFields.forEach(field => {
                          if (data[field] !== undefined) {
                              loraData[field] = data[field];
                          }
                      });
                      config = { ...config, ...loraData };
                  } else if (node.type === 'miscNode') {
                      // Only merge misc-specific fields
                      const miscFields = [
                          'resolution',
                          'train_batch_size',
                          'num_train_epochs',
                          'learning_rate',
                          'optimizer',
                          'mixed_precision',
                          'lr_scheduler',
                          'seed',
                          'caption_dropout',
                          'save_model_epochs',
                          'blocks_to_swap',
                          'gradient_checkpointing',
                          'recreate_cache',
                          'validation_epochs',
                          'validation_ratio',
                          'weighting_scheme',
                          'logit_mean',
                          'logit_std',
                          'mode_scale',
                          'guidance_scale',
                          'mask_dropout',

                          'freeze_transformer_layers',
                          'freeze_single_transformer_layers'
                      ];
                      const miscData = {};
                      miscFields.forEach(field => {
                          if (data[field] !== undefined) {
                              miscData[field] = data[field];
                          }
                      });
                      config = { ...config, ...miscData };
                  } else {
                      // For other nodes, merge all data
                      config = { ...config, ...data };
                  }
                  
                  console.log(`Config after merge:`, config);
              }
          }
      } else {
          console.log(`No edge found for handle ${handleId}`);
          
          // Fallback: Look for nodes of specific types if no edge is found
          let fallbackNode = null;
          switch(handleId) {
              case 'dir_out':
                  fallbackNode = nodes.find(n => n.type === 'directoryNode');
                  break;
              case 'lora_out':
                  fallbackNode = nodes.find(n => n.type === 'loraNode');
                  break;
              case 'misc_out':
                  fallbackNode = nodes.find(n => n.type === 'miscNode');
                  break;
          }
          
          if (fallbackNode) {
              console.log(`Fallback: Found ${handleId} node by type:`, fallbackNode);
              const { onChange, label, slots, ...data } = fallbackNode.data;
              console.log(`Merging fallback data from node ${fallbackNode.id}:`, data);
              
              // Selective merging for fallback as well
              if (fallbackNode.type === 'directoryNode') {
                  // Only merge directory-specific fields to avoid overwriting with defaults
                  const dirFields = [
                      'pretrained_model_name_or_path',
                      'output_dir',
                      'logging_dir',
                      'save_name',
                      'train_data_dir',
                      'model_path',
                      'vae_path'
                  ];
                  const dirData = {};
                  dirFields.forEach(field => {
                      if (data[field] !== undefined) {
                          dirData[field] = data[field];
                      }
                  });
                  config = { ...config, ...dirData };
              } else if (fallbackNode.type === 'loraNode') {
                  // Only merge LoRA-specific fields (excluding pretrained_model_name_or_path to avoid conflicts)
                  const loraFields = [
                      'rank',
                      'rank_alpha',
                      'use_lokr',
                      'lokr_factor',
                      'lora_layers'
                  ];
                  const loraData = {};
                  loraFields.forEach(field => {
                      if (data[field] !== undefined) {
                          loraData[field] = data[field];
                      }
                  });
                  config = { ...config, ...loraData };
              } else if (fallbackNode.type === 'miscNode') {
                  // Only merge misc-specific fields
                  const miscFields = [
                      'resolution',
                      'train_batch_size',
                      'num_train_epochs',
                      'learning_rate',
                      'optimizer',
                      'mixed_precision',
                      'lr_scheduler',
                      'seed',
                      'caption_dropout',
                      'save_model_epochs',
                      'blocks_to_swap',
                      'gradient_checkpointing',
                      'recreate_cache',
                      'validation_epochs',
                      'validation_ratio',
                      'weighting_scheme',
                      'logit_mean',
                      'logit_std',
                      'mode_scale',
                      'guidance_scale',
                      'mask_dropout',

                      'freeze_transformer_layers',
                      'freeze_single_transformer_layers'
                  ];
                  const miscData = {};
                  miscFields.forEach(field => {
                      if (data[field] !== undefined) {
                          miscData[field] = data[field];
                      }
                  });
                  config = { ...config, ...miscData };
              } else {
                  // For other nodes, merge all data
                  config = { ...config, ...data };
              }
              
              console.log(`Config after fallback merge:`, config);
          }
      }
  };

  console.log('Merging directory config');
  mergeFromSourceHandle('dir_out');
  console.log('Merging LoRA config');
  mergeFromSourceHandle('lora_out');
  console.log('Merging misc config');
  mergeFromSourceHandle('misc_out');
  console.log('Final config after merging globals:', config);

  // --- TRAVERSAL LOGIC FOR LISTS (Abstract -> Detail) ---

  // Helper: Get all items connected to a List Node's output slots
  // Structure: List (Source) -> [Edges] -> Items (Targets)
  const parseListItems = (listNodeId: string, itemProcessor: (node: Node) => any) => {
      const listNode = nodes.find(n => n.id === listNodeId);
      console.log(`parseListItems for node ${listNodeId}:`, listNode);
      if(!listNode || !listNode.data.slots) {
          console.log(`No list node or slots for ${listNodeId}`);
          return [];
      }
      
      console.log(`Slots for ${listNodeId}:`, listNode.data.slots);
      const results: any[] = [];
      
      listNode.data.slots.forEach((slotId: string) => {
          console.log(`Looking for edge from ${listNodeId} with handle ${slotId}`);
          // Find edge where Source is List and SourceHandle is Slot
          const edge = edges.find(e => e.source === listNodeId && e.sourceHandle === slotId);
          console.log(`Found edge:`, edge);
          if(edge) {
              const itemNode = nodes.find(n => n.id === edge.target);
              console.log(`Found item node:`, itemNode);
              if(itemNode) {
                  const processed = itemProcessor(itemNode);
                  console.log(`Processed item:`, processed);
                  if(processed) results.push(processed);
              }
          }
      });
      console.log(`Results for ${listNodeId}:`, results);
      return results;
  }

  // Helper: Find the List Node connected to a specific output handle of a Parent Node
  const getConnectedListResult = (parentNodeId: string, handleId: string, itemProcessor: (node: Node) => any) => {
      const edge = edges.find(e => e.source === parentNodeId && e.sourceHandle === handleId);
      if(!edge) return null;
      const result = parseListItems(edge.target, itemProcessor);
    console.log(`getConnectedListResult for ${parentNodeId}:${handleId} returned:`, result);
    return result;
  }

  // --- PARSE NEW SCHEMA (dataset_configs) ---
  console.log('Looking for dataset edge from script node');
  const dsEdge = edges.find(e => e.source === scriptNode.id && e.sourceHandle === 'json_out');
  console.log('Dataset edge found:', dsEdge);
  
  // Fallback: Look for dataset nodes by type if no edge is found
  let fallbackDsNode = null;
  let isFallback = false;
  if (!dsEdge) {
      console.log('No dataset edge found, looking for dataset nodes by type');
      fallbackDsNode = nodes.find(n => n.type === 'datasetListNode' || n.type === 'datasetConfigNode');
      if (fallbackDsNode) {
          console.log('Found fallback dataset node:', fallbackDsNode);
          isFallback = true;
      }
  }
  
  const targetNode = dsEdge ? nodes.find(n => n.id === dsEdge.target) : fallbackDsNode;
  console.log('Target node for dataset:', targetNode);
  
  if (targetNode && (targetNode.type === 'datasetListNode' || targetNode.type === 'datasetConfigNode')) {
      config.dataset_configs = [];
        
        const buildDataset = (dsNode: Node) => {
            console.log('Building dataset from node:', dsNode);
            const d: any = {
                train_data_dir: dsNode.data.train_data_dir,
                resolution: dsNode.data.resolution || config.resolution,
                recreate_cache: false,
                repeats: dsNode.data.repeats || 1,
            };
            console.log('Dataset node data:', dsNode.data);
            if(dsNode.data.recreate_cache_label) d.recreate_cache_label = true;
            if(dsNode.data.recreate_cache_target) d.recreate_cache_target = true;
            if(dsNode.data.recreate_cache_reference) d.recreate_cache_reference = true;
            if(dsNode.data.recreate_cache_caption) d.recreate_cache_caption = true;

            // Only try to get connected list items if this is not a fallback situation
            // In fallback mode, we assume the node is standalone and contains all its data
            if (!isFallback) {
                const imgs = getConnectedListResult(dsNode.id, 'image_config', (n) => (n.data.key && n.data.suffix !== undefined) ? { [n.data.key]: { suffix: n.data.suffix } } : null);
                if(imgs && imgs.length > 0) d.image_configs = Object.assign({}, ...imgs);

                const targets = getConnectedListResult(dsNode.id, 'target_config', (n) => (n.data.key) ? { key: n.data.key, val: { image: n.data.image, from_image: n.data.from_image } } : null);
                if(targets && targets.length > 0) {
                    d.target_configs = {};
                    targets.forEach((t: any) => {
                        if(!d.target_configs[t.key]) d.target_configs[t.key] = [];
                        const obj: any = { image: t.val.image };
                        if(t.val.from_image) obj.from_image = t.val.from_image;
                        d.target_configs[t.key].push(obj);
                    });
                }

                const refs = getConnectedListResult(dsNode.id, 'reference_config', (refGroupNode) => {
                        if (!refGroupNode.data.key) return null;
                        const entries = parseListItems(refGroupNode.id, (entryNode) => {
                            const e = { ...entryNode.data };
                            delete e.label;
                            if(e.sample_type === 'from_same_name') return { sample_type: 'from_same_name', image: e.image };
                            return { sample_type: e.sample_type, image: e.image, suffix: e.suffix, resize: e.resize, count: e.count };
                        });
                        return (entries.length > 0) ? { key: refGroupNode.data.key, entries: entries } : null;
                });
                if(refs && refs.length > 0) {
                    d.reference_configs = {};
                    refs.forEach((r: any) => {
                            if(!d.reference_configs[r.key]) d.reference_configs[r.key] = [];
                            d.reference_configs[r.key] = [...d.reference_configs[r.key], ...r.entries];
                    });
                }

                const caps = getConnectedListResult(dsNode.id, 'caption_config', (n) => {
                    if(!n.data.key) return null;
                    const obj: any = { ext: n.data.ext, image: n.data.image };
                    if(n.data.reference_list) obj.reference_list = n.data.reference_list;
                    return { [n.data.key]: obj };
                });
                if(caps && caps.length > 0) d.caption_configs = Object.assign({}, ...caps);

                const batches = getConnectedListResult(dsNode.id, 'batch_config', (n) => {
                        const clean = { target_config: n.data.target_config, caption_config: n.data.caption_config, caption_dropout: n.data.caption_dropout };
                        if(n.data.reference_config) (clean as any).reference_config = n.data.reference_config;
                        return clean;
                });
                if(batches && batches.length > 0) d.batch_configs = batches;
            }
            return d;
        }

        if (targetNode.type === 'datasetListNode') {
            // For fallback, just create a simple dataset config
            if (isFallback) {
                const dsNode = targetNode;
                const d: any = {
                    train_data_dir: dsNode.data.train_data_dir,
                    resolution: dsNode.data.resolution || config.resolution,
                    recreate_cache: false,
                    repeats: dsNode.data.repeats || 1,
                };
                console.log('Dataset node data (fallback):', dsNode.data);
                if(dsNode.data.recreate_cache_target) d.recreate_cache_target = true;
                if(dsNode.data.recreate_cache_reference) d.recreate_cache_reference = true;
                if(dsNode.data.recreate_cache_caption) d.recreate_cache_caption = true;
                config.dataset_configs.push(d);
            } else {
                config.dataset_configs = parseListItems(targetNode.id, buildDataset);
            }
        } else if (targetNode.type === 'datasetConfigNode') {
            // For fallback, just create a simple dataset config
            if (isFallback) {
                const dsNode = targetNode;
                const d: any = {
                    train_data_dir: dsNode.data.train_data_dir,
                    resolution: dsNode.data.resolution || config.resolution,
                    recreate_cache: false,
                    repeats: dsNode.data.repeats || 1,
                };
                console.log('Dataset node data (fallback):', dsNode.data);
                if(dsNode.data.recreate_cache_target) d.recreate_cache_target = true;
                if(dsNode.data.recreate_cache_reference) d.recreate_cache_reference = true;
                if(dsNode.data.recreate_cache_caption) d.recreate_cache_caption = true;
                config.dataset_configs.push(d);
            } else {
                config.dataset_configs.push(buildDataset(targetNode));
            }
        }
    
  // --- PARSE OLD SCHEMA (root merge) ---
  const oldEdge = edges.find(e => e.source === scriptNode.id && e.sourceHandle === 'old_out');
    
  // Fallback: Look for old schema root by type if no edge is found
  let fallbackOldNode = null;
  if (!oldEdge) {
      console.log('No old schema edge found, looking for old schema root by type');
      fallbackOldNode = nodes.find(n => n.type === 'oldSchemaRoot');
      if (fallbackOldNode) {
          console.log('Found fallback old schema node:', fallbackOldNode);
      }
  }
    
  const oldTargetNode = oldEdge ? nodes.find(n => n.id === oldEdge.target) : fallbackOldNode;
  
  if (oldTargetNode && oldTargetNode.type === 'oldSchemaRoot') {
        
        // A. Image Configs
        const imgConfigs = getConnectedListResult(oldTargetNode.id, 'image_configs', (n) => {
            return (n.data.key !== undefined) ? { [n.data.key]: { suffix: n.data.suffix } } : null;
        });
        if(imgConfigs && imgConfigs.length > 0) config.image_configs = Object.assign({}, ...imgConfigs);

        // B. Caption Configs
        const capConfigs = getConnectedListResult(oldTargetNode.id, 'caption_configs', (n) => {
            if(!n.data.key) return null;
            const obj: any = {};
            if(n.data.ext) obj.ext = n.data.ext;
            if(n.data.instruction) obj.instruction = n.data.instruction;
            
            // Check for Nested Reference List (Caption Item -> Ref List -> Ref Item)
            const refListItems = getConnectedListResult(n.id, 'ref_list', (refItem) => {
                 const r: any = { target: refItem.data.target };
                 if(refItem.data.resize) r.resize = refItem.data.resize;
                 if(refItem.data.dropout !== undefined) r.dropout = refItem.data.dropout;
                 return r;
            });
            
            if(refListItems && refListItems.length > 0) {
                obj.ref_image_config = { ref_image_list: refListItems };
            }
            return { [n.data.key]: obj };
        });
        if(capConfigs && capConfigs.length > 0) config.caption_configs = Object.assign({}, ...capConfigs);

        // C. Training Set
        const trainingSet = getConnectedListResult(oldTargetNode.id, 'training_set', (n) => {
            const setObj: any = {};
            if(n.data.captions_selection_target) {
                setObj.captions_selection = { target: n.data.captions_selection_target };
            }

            // Nested Layouts (Training Set Item -> Layout List -> Layout Item)
            const layouts = getConnectedListResult(n.id, 'layout_list', (l) => {
                if(!l.data.key) return null;
                const lo: any = { target: l.data.target };
                if(l.data.noised) lo.noised = true;
                if(l.data.dropout !== undefined) lo.dropout = l.data.dropout;
                return { [l.data.key]: lo };
            });

            if(layouts && layouts.length > 0) {
                setObj.training_layout_configs = Object.assign({}, ...layouts);
            }
            return setObj;
        });
        if(trainingSet && trainingSet.length > 0) config.training_set = trainingSet;
    }
  }

  console.log('Final config:', config);
  const cmd = `python ${scriptName} --config_path ${configPath}`;
  console.log('Returning result:', { config, command: cmd });
  return { config, command: cmd };
};
