import gradio as gr
import subprocess
import json
import sys
import os
import glob

# ===== 语言翻译系统 =====
TRANSLATIONS = {
    'zh': {
        'title': '## LoRA 训练',
        'script': '训练脚本',
        'config_path': '配置文件路径 (.json文件)',
        'config_path_placeholder': '输入保存/加载配置的路径',
        'save': '保存',
        'load': '加载',
        'directory_section': '目录配置',
        'output_dir': '输出目录',
        'output_dir_placeholder': '检查点保存位置',
        'save_name': '保存名称',
        'save_name_placeholder': '检查点保存名称',
        'pretrained_model_name_or_path': '预训练模型名称或路径',
        'pretrained_model_placeholder': '仓库名称或包含diffusers模型结构的目录',
        'resume_from_checkpoint': '从检查点恢复',
        'resume_checkpoint_placeholder': '从选定目录恢复lora权重',
        'train_data_dir': '训练数据目录',
        'train_data_dir_placeholder': '包含数据集的目录',
        'model_path': '模型路径',
        'model_path_placeholder': '如果不是从官方权重训练则为单个权重文件',
        'report_to': '报告到',
        'lora_config': 'LoRA 配置',
        'rank': '秩',
        'rank_info': '建议对小于100的训练集使用秩4',
        'train_batch_size': '训练批次大小',
        'batch_size_info': '批次大小1使用18GB。请使用小批次大小以避免内存不足',
        'repeats': '重复次数',
        'gradient_accumulation_steps': '梯度累积步数',
        'mixed_precision': '混合精度',
        'gradient_checkpointing': '梯度检查点',
        'optimizer': '优化器',
        'lr_scheduler': '学习率调度器',
        'cosine_restarts': '余弦重启',
        'cosine_restarts_info': '仅对学习率调度器cosine_with_restarts有用',
        'learning_rate': '学习率',
        'learning_rate_info': '推荐：1e-4 或 prodigy使用1',
        'lr_warmup_steps': '学习率预热步数',
        'seed': '随机种子',
        'blocks_to_swap': '交换块数',
        'blocks_to_swap_info': '交换到CPU的块数。建议24GB使用10，更低显存使用更多',
        'mask_dropout': '掩码丢弃',
        'mask_dropout_info': '丢弃掩码，意味着整个图像重建的掩码全为1',
        'reg_ratio': '正则化比率',
        'reg_ratio_info': '作为目标迁移学习的正则化。如果不训练不同目标则设为1',
        'reg_timestep': '正则化时间步',
        'reg_timestep_info': '作为目标迁移学习的正则化。如果不训练不同目标则设为0',
        'misc': '杂项',
        'num_train_epochs': '训练轮数',
        'num_train_epochs_info': '训练的总轮数',
        'save_model_epochs': '保存模型轮数',
        'save_model_epochs_info': '每x轮保存检查点',
        'validation_epochs': '验证轮数',
        'validation_epochs_info': '每x轮执行验证',
        'skip_epoch': '跳过轮数',
        'skip_epoch_info': '跳过x轮进行验证和保存检查点',
        'skip_step': '跳过步数',
        'skip_step_info': '跳过x步进行验证和保存检查点',
        'validation_ratio': '验证比例',
        'validation_ratio_info': '按此比例分割数据集用于验证',
        'recreate_cache': '重新创建缓存',
        'caption_dropout': '标题丢弃',
        'caption_dropout_info': '标题丢弃',
        'max_time_steps': '最大时间步限制',
        'max_time_steps_info': '最大时间步限制',
        # 'resolution_section': '## 实验选项：分辨率\n- 基于目标分辨率（默认：1024）。\n- 支持512或1024。',
        'resolution': '分辨率',
        'output_box': '输出框',
        'run': '运行',
        'language_toggle': '🌐 切换到English',
        'slider': '滑块训练相关',
        'use_two_captions': '使用两对应文本标注',
        'slider_positive_scale': '滑块正向目标强度',
        'slider_negative_scale': '滑块负面目标强度',
        'config_template': '训练排版配置模板(Kontext_new & qwen image)',
        'select_template': '选择配置模板',
        'save_edited_json': '保存编辑后的JSON到Config'
    },
    'en': {
        'title': '## Lora Training',
        'script': 'script',
        'config_path': 'Config Path (.json file)',
        'config_path_placeholder': 'Enter path to save/load config',
        'save': 'Save',
        'load': 'load',
        'directory_section': 'Directory section',
        'output_dir': 'output_dir',
        'output_dir_placeholder': 'checkpoint save to',
        'save_name': 'save_name',
        'save_name_placeholder': 'checkpoint save name',
        'pretrained_model_name_or_path': 'pretrained_model_name_or_path',
        'pretrained_model_placeholder': 'repo name or dir contains diffusers model structure',
        'resume_from_checkpoint': 'resume_from_checkpoint',
        'resume_checkpoint_placeholder': 'resume the lora weight from seleted dir',
        'train_data_dir': 'train_data_dir',
        'train_data_dir_placeholder': 'dir contains dataset',
        'model_path': 'model_path',
        'model_path_placeholder': 'single weight files if not trained from official weight',
        'report_to': 'report_to',
        'lora_config': 'Lora Config',
        'rank': 'rank',
        'rank_info': 'Recommanded to use rank 4 for training set small than 100.',
        'train_batch_size': 'train_batch_size',
        'batch_size_info': 'Batch size 1 is using 18GB. Please use small batch size to avoid oom.',
        'repeats': 'repeats',
        'gradient_accumulation_steps': 'gradient_accumulation_steps',
        'mixed_precision': 'mixed_precision',
        'gradient_checkpointing': 'gradient_checkpointing',
        'optimizer': 'optimizer',
        'lr_scheduler': 'lr_scheduler',
        'cosine_restarts': 'cosine_restarts',
        'cosine_restarts_info': 'Only useful for lr_scheduler: cosine_with_restarts',
        'learning_rate': 'learning_rate',
        'learning_rate_info': 'Recommended: 1e-4 or 1 for prodigy',
        'lr_warmup_steps': 'lr_warmup_steps',
        'seed': 'seed',
        'blocks_to_swap': 'blocks_to_swap',
        'blocks_to_swap_info': 'How many blocks to swap to cpu. It is suggested 10 for 24 GB and more for lower VRAM',
        'mask_dropout': 'mask_dropout',
        'mask_dropout_info': 'Dropout mask which means mask is all one for whole image reconstruction',
        'reg_ratio': 'reg_ratio',
        'reg_ratio_info': 'As regularization of objective transfer learning. Set as 1 if you aren\'t training different objective.',
        'reg_timestep': 'reg_timestep',
        'reg_timestep_info': 'As regularization of objective transfer learning. Set as 0 if you aren\'t training different objective.',
        'misc': 'Misc',
        'num_train_epochs': 'num_train_epochs',
        'num_train_epochs_info': 'Total epoches of the training',
        'save_model_epochs': 'save_model_epochs',
        'save_model_epochs_info': 'Save checkpoint when x epoches',
        'validation_epochs': 'validation_epochs',
        'validation_epochs_info': 'perform validation when x epoches',
        'skip_epoch': 'skip_epoch',
        'skip_epoch_info': 'Skip x epoches for validation and save checkpoint',
        'skip_step': 'skip_step',
        'skip_step_info': 'Skip x steps for validation and save checkpoint',
        'validation_ratio': 'validation_ratio',
        'validation_ratio_info': 'Split dataset with this ratio for validation',
        'recreate_cache': 'recreate_cache',
        'caption_dropout': 'Caption Dropout',
        'caption_dropout_info': 'Caption Dropout',
        'max_time_steps': 'Max timesteps limitation',
        'max_time_steps_info': 'Max timesteps limitation',
        # 'resolution_section': '## Experiment Option: resolution\n- Based target resolution (default:1024). \n- 512 or 1024 are supported.',
        'resolution': 'resolution',
        'output_box': 'Output Box',
        'run': 'Run',
        'language_toggle': '🌐 切换到中文',
        'slider': 'Slider Related',
        'use_two_captions': 'Use two captions for each direction',
        'slider_positive_scale': 'Slider positive scale',
        'slider_negative_scale': 'Slider negative scale',
        'config_template': 'Train Layout Config Template (Kontext_new & qwen image)',
        'select_template': 'Select config template',
        'save_edited_json': 'Save Edited JSON to Config'
    }
}

current_language = 'en'

def update_labels_only():
    toggle_language()
    return {
        title_md: gr.update(value=get_text('title')),
        script: gr.update(label=get_text('script')),
        config_path: gr.update(label=get_text('config_path'), placeholder=get_text('config_path_placeholder')),
        save_config_btn: gr.update(value=get_text('save')),
        load_config_btn: gr.update(value=get_text('load')),
        output_dir: gr.update(label=get_text('output_dir'), placeholder=get_text('output_dir_placeholder')),
        save_name: gr.update(label=get_text('save_name'), placeholder=get_text('save_name_placeholder')),
        pretrained_model_name_or_path: gr.update(label=get_text('pretrained_model_name_or_path'), placeholder=get_text('pretrained_model_placeholder')),
        resume_from_checkpoint: gr.update(label=get_text('resume_from_checkpoint'), placeholder=get_text('resume_checkpoint_placeholder')),
        train_data_dir: gr.update(label=get_text('train_data_dir'), placeholder=get_text('train_data_dir_placeholder')),
        model_path: gr.update(label=get_text('model_path'), placeholder=get_text('model_path_placeholder')),
        report_to: gr.update(label=get_text('report_to')),
        rank: gr.update(label=get_text('rank'), info=get_text('rank_info')),
        train_batch_size: gr.update(label=get_text('train_batch_size'), info=get_text('batch_size_info')),
        repeats: gr.update(label=get_text('repeats')),
        gradient_accumulation_steps: gr.update(label=get_text('gradient_accumulation_steps')),
        mixed_precision: gr.update(label=get_text('mixed_precision')),
        gradient_checkpointing: gr.update(label=get_text('gradient_checkpointing')),
        optimizer: gr.update(label=get_text('optimizer')),
        lr_scheduler: gr.update(label=get_text('lr_scheduler')),
        cosine_restarts: gr.update(label=get_text('cosine_restarts'), info=get_text('cosine_restarts_info')),
        learning_rate: gr.update(label=get_text('learning_rate'), info=get_text('learning_rate_info')),
        lr_warmup_steps: gr.update(label=get_text('lr_warmup_steps')),
        seed: gr.update(label=get_text('seed')),
        blocks_to_swap: gr.update(label=get_text('blocks_to_swap'), info=get_text('blocks_to_swap_info')),
        mask_dropout: gr.update(label=get_text('mask_dropout'), info=get_text('mask_dropout_info')),
        reg_ratio: gr.update(label=get_text('reg_ratio'), info=get_text('reg_ratio_info')),
        reg_timestep: gr.update(label=get_text('reg_timestep'), info=get_text('reg_timestep_info')),
        num_train_epochs: gr.update(label=get_text('num_train_epochs'), info=get_text('num_train_epochs_info')),
        save_model_epochs: gr.update(label=get_text('save_model_epochs'), info=get_text('save_model_epochs_info')),
        validation_epochs: gr.update(label=get_text('validation_epochs'), info=get_text('validation_epochs_info')),
        skip_epoch: gr.update(label=get_text('skip_epoch'), info=get_text('skip_epoch_info')),
        skip_step: gr.update(label=get_text('skip_step'), info=get_text('skip_step_info')),
        validation_ratio: gr.update(label=get_text('validation_ratio'), info=get_text('validation_ratio_info')),
        recreate_cache: gr.update(label=get_text('recreate_cache')),
        caption_dropout: gr.update(label=get_text('caption_dropout'), info=get_text('caption_dropout_info')),
        max_time_steps: gr.update(label=get_text('max_time_steps'), info=get_text('max_time_steps_info')),
        resolution: gr.update(label=get_text('resolution')),
        use_two_captions: gr.update(label=get_text('use_two_captions')),
        slider_positive_scale: gr.update(label=get_text('slider_positive_scale')),
        slider_negative_scale: gr.update(label=get_text('slider_negative_scale')),
        template_dropdown: gr.update(label=get_text('select_template')),
        # save_edited_json_btn: gr.update(value=get_text('save_edited_json')),
        language_toggle_btn: gr.update(value=get_text('language_toggle')),
        output: gr.update(label=get_text('output_box')),
        run_btn: gr.update(value=get_text('run')),
        
        save_config_btn2: gr.update(value=get_text('save')),
        load_config_btn2: gr.update(value=get_text('load')),
    }

def get_text(key):
    return TRANSLATIONS[current_language].get(key, key)

def toggle_language():
    global current_language
    current_language = 'en' if current_language == 'zh' else 'zh'

TEMPLATE_DIR = "./config_template"
os.makedirs(TEMPLATE_DIR, exist_ok=True)

def list_templates(config_path):
    # return [os.path.basename(f) for f in glob.glob(os.path.join(TEMPLATE_DIR, "*.json"))]
    templates = [os.path.basename(f) for f in glob.glob(os.path.join(TEMPLATE_DIR, "*.json"))]
    templates.sort()
    # 把 config_path 也加进去（只保留文件名，便于显示）
    config_name = os.path.basename(config_path) if os.path.isfile(config_path) else "config.json"
    choices = [config_name] + templates
    return choices
# ------------------------------------------------------------------
#  Script-dependent template injection
# ------------------------------------------------------------------
def change_script(selected_script, config_path):
    """
    Called when the script dropdown changes.
    If the chosen script is 'train_qwen_image.py' we ensure that
    'image_configs', 'caption_configs', and 'training_set' exist
    by merging them from preset_0_single.json when they are absent.
    The merged (or original) config is then returned for display.
    """
    if selected_script != "train_qwen_image.py" and selected_script != "train_flux_lora_ui_kontext_new.py":
        return ""
    preset_name = "preset_0_single.json"
    preset_path = os.path.join(TEMPLATE_DIR, preset_name)

    editor = {}
    preset = {}
    mandatory = {"image_configs", "caption_configs", "training_set"}
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                preset = json.load(f)
        except Exception:
            pass  # ignore read/parse errors
    else:
        try:
            with open(preset_path, "r", encoding="utf-8") as f:
                preset = json.load(f)
        except Exception:
            # Preset unreadable → return current config
            # return json.dumps(current, indent=4, ensure_ascii=False)
            print(f"Error reading {preset_path}")

    # 5. Merge missing mandatory sections from preset
    for key in mandatory:
        if key in preset:
            editor[key] = preset[key]

    # 6. Return the merged JSON for display
    return json.dumps(editor, indent=4, ensure_ascii=False)

def load_template(template_name, config_path):
    """
    仅当 config_path 中缺少 image_configs / caption_configs / training_set 时，
    才用模板里的对应键补充，其余保持不变。
    """
    if isinstance(template_name, list):
        template_name = template_name[0] if template_name else ""
    if not template_name:
        return ""

    tpl = {}
    config_name = os.path.basename(config_path)
    if template_name == config_name and os.path.isfile(config_path):
        # 选的是当前 config
        with open(config_path, "r", encoding="utf-8") as f:
            tpl = json.load(f)
    else:
        # 选的是模板
        tpl_path = os.path.join(TEMPLATE_DIR, template_name)
        if not os.path.isfile(tpl_path):
            return ""

        with open(tpl_path, "r", encoding="utf-8") as f:
            tpl = json.load(f)
        
    current = {}

    # 仅补充缺失的三项
    for key in ("image_configs", "caption_configs", "training_set"):
        if key in tpl and key not in current:
            current[key] = tpl[key]

    return json.dumps(current, indent=4, ensure_ascii=False)


default_config = {
    "script": "train_flux_lora_ui_kontext.py",
    "script_choices": [
        "train_qwen_image_edit.py",
        "train_qwen_image.py",
        "train_flux_lora_ui_kontext_new.py",
        "train_flux_lora_ui_kontext.py",
        "train_flux_lora_ui_kontext_slider.py",
        "train_flux_lora_ui_with_mask.py",
        "train_flux_lora_ui.py",
    ],
    "output_dir": "/home/waas/kontext_output",
    "save_name": "flux-lora",
    "pretrained_model_name_or_path": "/datasets/T2ITrainer/models/flux-kontext-nf4",
    "train_data_dir": "/home/waas/kontext_test",
    "resume_from_checkpoint": None,
    "model_path": None,
    "report_to": "all",
    "rank": 16,
    "train_batch_size": 1,
    "repeats": 1,
    "gradient_accumulation_steps": 1,
    "mixed_precision": "bf16",
    "gradient_checkpointing": True,
    "optimizer": "adamw",
    "lr_scheduler": "constant",
    "learning_rate": 1e-4,
    "lr_warmup_steps": 0,
    "seed": 4321,
    "num_train_epochs": 5,
    "save_model_epochs": 1,
    "validation_epochs": 1,
    "skip_epoch": 0,
    "skip_step": 0,
    "validation_ratio": 0.1,
    "recreate_cache": False,
    "caption_dropout": 0.1,
    "config_path": "config.json",
    "resolution": "512",
    "resolution_choices": ["1344","1024", "768", "512", "256"],
    "use_debias": False,
    "snr_gamma": 0,
    "cosine_restarts": 1,
    "max_time_steps": 0,
    "blocks_to_swap": 0,
    "mask_dropout": 0,
    "reg_ratio": 0.0,
    "reg_timestep": 0,
    "use_two_captions": False,
    "slider_positive_scale": 1.0,
    "slider_negative_scale": -1.0
}

def save_config(
    config_path, script, seed, mixed_precision, report_to, lr_warmup_steps,
    output_dir, save_name, train_data_dir, optimizer, lr_scheduler, learning_rate,
    train_batch_size, repeats, gradient_accumulation_steps, num_train_epochs,
    save_model_epochs, validation_epochs, rank, skip_epoch, skip_step,
    gradient_checkpointing, validation_ratio, pretrained_model_name_or_path,
    model_path, resume_from_checkpoint, recreate_cache, resolution,
    caption_dropout, cosine_restarts, max_time_steps, blocks_to_swap,
    mask_dropout, reg_ratio, reg_timestep, use_two_captions,
    slider_positive_scale, slider_negative_scale,
    json_editor        # 这里只放 json_editor 字符串
):
    
    # 1️⃣ 先读原文件
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = {}

    # 2️⃣ UI 实时值覆盖
    ui_vals = {
        "config_path": config_path,
        "script": script,
        "seed": seed,
        "mixed_precision": mixed_precision,
        "report_to": report_to,
        "lr_warmup_steps": lr_warmup_steps,
        "output_dir": output_dir,
        "save_name": save_name,
        "train_data_dir": train_data_dir,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "learning_rate": learning_rate,
        "train_batch_size": train_batch_size,
        "repeats": repeats,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_train_epochs": num_train_epochs,
        "save_model_epochs": save_model_epochs,
        "validation_epochs": validation_epochs,
        "rank": rank,
        "skip_epoch": skip_epoch,
        "skip_step": skip_step,
        "gradient_checkpointing": gradient_checkpointing,
        "validation_ratio": validation_ratio,
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
        "model_path": model_path,
        "resume_from_checkpoint": resume_from_checkpoint,
        "recreate_cache": recreate_cache,
        "resolution": resolution,
        "caption_dropout": caption_dropout,
        "cosine_restarts": cosine_restarts,
        "max_time_steps": max_time_steps,
        "blocks_to_swap": blocks_to_swap,
        "mask_dropout": mask_dropout,
        "reg_ratio": reg_ratio,
        "reg_timestep": reg_timestep,
        "use_two_captions": use_two_captions,
        "slider_positive_scale": slider_positive_scale,
        "slider_negative_scale": slider_negative_scale,
    }
    cfg.update(ui_vals)

    # 3️⃣ json_editor 只做「缺键补全」
    try:
        editor_partial = json.loads(json_editor)
    except json.JSONDecodeError:
        editor_partial = {}
        
    # for k, v in editor_partial.items():
    #     cfg.setdefault(k, v)
    
    for key in ("image_configs", "caption_configs", "training_set"):
        if key in editor_partial:
            cfg[key] = {}
            cfg[key] = editor_partial[key]

    print(f"save config to {config_path}")
    # 4️⃣ 写回
    os.makedirs(os.path.dirname(config_path) or ".", exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)
    # 同时保留默认副本
    with open("config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)

# 启动时默认加载 config
def load_first_template():
    templates = list_templates(default_config["config_path"])
    default_choice = os.path.basename(default_config["config_path"])
    return templates, default_choice, load_template(default_choice, default_config["config_path"])


def load_config(config_path):
    if not os.path.isfile(config_path):
        save_config(**default_config)
    with open(config_path, 'r') as f:
        config = json.load(f)
    for k in config:
        default_config[k] = config[k]
    
    loaded_config = [config.get(k, default_config[k]) for k in [
        "config_path","script","seed","mixed_precision","report_to","lr_warmup_steps",
        "output_dir","save_name","train_data_dir","optimizer","lr_scheduler","learning_rate",
        "train_batch_size","repeats","gradient_accumulation_steps","num_train_epochs",
        "save_model_epochs","validation_epochs","rank","skip_epoch","skip_step",
        "gradient_checkpointing","validation_ratio","pretrained_model_name_or_path",
        "model_path","resume_from_checkpoint","recreate_cache","resolution",
        "caption_dropout","cosine_restarts","max_time_steps","blocks_to_swap",
        "mask_dropout","reg_ratio","reg_timestep","use_two_captions",
        "slider_positive_scale","slider_negative_scale"
    ]]
    
    _, _, first_template = load_first_template()
    
    loaded_config += [first_template]
    
    return loaded_config

load_config("config.json")

def run(
        config_path, script, seed, mixed_precision, report_to, lr_warmup_steps,
        output_dir, save_name, train_data_dir, optimizer, lr_scheduler, learning_rate,
        train_batch_size, repeats, gradient_accumulation_steps, num_train_epochs,
        save_model_epochs, validation_epochs, rank, skip_epoch, skip_step,
        gradient_checkpointing, validation_ratio, pretrained_model_name_or_path,
        model_path, resume_from_checkpoint, recreate_cache, resolution,
        caption_dropout, cosine_restarts, max_time_steps, blocks_to_swap,
        mask_dropout, reg_ratio, reg_timestep, use_two_captions,
        slider_positive_scale, slider_negative_scale,
        json_editor
    ):
    save_config(config_path, script, seed, mixed_precision, report_to, lr_warmup_steps,
        output_dir, save_name, train_data_dir, optimizer, lr_scheduler, learning_rate,
        train_batch_size, repeats, gradient_accumulation_steps, num_train_epochs,
        save_model_epochs, validation_epochs, rank, skip_epoch, skip_step,
        gradient_checkpointing, validation_ratio, pretrained_model_name_or_path,
        model_path, resume_from_checkpoint, recreate_cache, resolution,
        caption_dropout, cosine_restarts, max_time_steps, blocks_to_swap,
        mask_dropout, reg_ratio, reg_timestep, use_two_captions,
        slider_positive_scale, slider_negative_scale,
        json_editor)
    cmd = [sys.executable, script, "--config_path", config_path]
    subprocess.call(cmd)
    return " ".join(cmd)

# def update_language_interface():
#     toggle_language()
#     return [get_text('title'),
#             get_text('language_toggle'),
#             get_text('select_template'),
#             get_text('save_edited_json')] + \
#            [get_text(k) for k in [
#                'script','config_path','config_path_placeholder','save','load',
#                'directory_section','output_dir','output_dir_placeholder',
#                'save_name','save_name_placeholder','pretrained_model_name_or_path',
#                'pretrained_model_placeholder','resume_from_checkpoint',
#                'resume_checkpoint_placeholder','train_data_dir','train_data_dir_placeholder',
#                'model_path','model_path_placeholder','report_to','lora_config','rank',
#                'rank_info','train_batch_size','batch_size_info','repeats',
#                'gradient_accumulation_steps','mixed_precision','gradient_checkpointing',
#                'optimizer','lr_scheduler','cosine_restarts','cosine_restarts_info',
#                'learning_rate','learning_rate_info','lr_warmup_steps','seed',
#                'blocks_to_swap','blocks_to_swap_info','mask_dropout','mask_dropout_info',
#                'reg_ratio','reg_ratio_info','reg_timestep','reg_timestep_info','misc',
#                'num_train_epochs','num_train_epochs_info','save_model_epochs',
#                'save_model_epochs_info','validation_epochs','validation_epochs_info',
#                'skip_epoch','skip_epoch_info','skip_step','skip_step_info',
#                'validation_ratio','validation_ratio_info','recreate_cache',
#                'caption_dropout','caption_dropout_info','max_time_steps',
#                'max_time_steps_info','resolution_section','resolution','output_box','run',
#                'slider','use_two_captions','slider_positive_scale','slider_negative_scale'
#            ]]

with gr.Blocks() as demo:
    with gr.Row():
        gr.HTML("<div style='flex-grow: 1;'></div>")
        language_toggle_btn = gr.Button(get_text('language_toggle'), scale=0, size="sm")
    title_md = gr.Markdown(get_text('title'))

    script = gr.Dropdown(label=get_text('script'), value=default_config["script"], choices=default_config["script_choices"])
    with gr.Row(equal_height=True):
        config_path = gr.Textbox(scale=3, label=get_text('config_path'), value=default_config["config_path"], placeholder=get_text('config_path_placeholder'))
        save_config_btn = gr.Button(get_text('save'), scale=1)
        load_config_btn = gr.Button(get_text('load'), scale=1)

    directory_accordion = gr.Accordion(get_text('directory_section'))
    with directory_accordion:
        with gr.Row():
            output_dir = gr.Textbox(label=get_text('output_dir'), value=default_config["output_dir"], placeholder=get_text('output_dir_placeholder'))
            save_name = gr.Textbox(label=get_text('save_name'), value=default_config["save_name"], placeholder=get_text('save_name_placeholder'))
        with gr.Row():
            pretrained_model_name_or_path = gr.Textbox(label=get_text('pretrained_model_name_or_path'), value=default_config["pretrained_model_name_or_path"], placeholder=get_text('pretrained_model_name_or_path_placeholder'))
            resume_from_checkpoint = gr.Textbox(label=get_text('resume_from_checkpoint'), value=default_config["resume_from_checkpoint"], placeholder=get_text('resume_checkpoint_placeholder'))
        with gr.Row():
            train_data_dir = gr.Textbox(label=get_text('train_data_dir'), value=default_config["train_data_dir"], placeholder=get_text('train_data_dir_placeholder'))
            model_path = gr.Textbox(label=get_text('model_path'), value=default_config["model_path"], placeholder=get_text('model_path_placeholder'))
        with gr.Row():
            report_to = gr.Dropdown(label=get_text('report_to'), value=default_config["report_to"], choices=["all","wandb","tensorboard"])

    lora_accordion = gr.Accordion(get_text('lora_config'))
    with lora_accordion:
        with gr.Row():
            rank = gr.Number(label=get_text('rank'), value=default_config["rank"], info=get_text('rank_info'))
            train_batch_size = gr.Number(label=get_text('train_batch_size'), value=default_config["train_batch_size"], info=get_text('batch_size_info'))
        with gr.Row():
            repeats = gr.Number(label=get_text('repeats'), value=default_config["repeats"])
            gradient_accumulation_steps = gr.Number(label=get_text('gradient_accumulation_steps'), value=default_config["gradient_accumulation_steps"])
            mixed_precision = gr.Radio(label=get_text('mixed_precision'), value=default_config["mixed_precision"], choices=["bf16","fp8"])
            gradient_checkpointing = gr.Checkbox(label=get_text('gradient_checkpointing'), value=default_config["gradient_checkpointing"])
        with gr.Row():
            optimizer = gr.Dropdown(label=get_text('optimizer'), value=default_config["optimizer"], choices=["adamw","prodigy"])
            lr_scheduler = gr.Dropdown(label=get_text('lr_scheduler'), value=default_config["lr_scheduler"], choices=["linear","cosine","cosine_with_restarts","polynomial","constant","constant_with_warmup"])
            cosine_restarts = gr.Number(label=get_text('cosine_restarts'), value=default_config["cosine_restarts"], info=get_text('cosine_restarts_info'), minimum=1)
        with gr.Row():
            learning_rate = gr.Number(label=get_text('learning_rate'), value=default_config["learning_rate"], info=get_text('learning_rate_info'))
            lr_warmup_steps = gr.Number(label=get_text('lr_warmup_steps'), value=default_config["lr_warmup_steps"])
            seed = gr.Number(label=get_text('seed'), value=default_config["seed"])
        with gr.Row():
            blocks_to_swap = gr.Number(label=get_text('blocks_to_swap'), value=default_config["blocks_to_swap"], info=get_text('blocks_to_swap_info'))
            mask_dropout = gr.Number(label=get_text('mask_dropout'), value=default_config["mask_dropout"], info=get_text('mask_dropout_info'))
            reg_ratio = gr.Number(label=get_text('reg_ratio'), value=default_config["reg_ratio"], info=get_text('reg_ratio_info'))
            reg_timestep = gr.Number(label=get_text('reg_timestep'), value=default_config["reg_timestep"], info=get_text('reg_timestep_info'))

    misc_accordion = gr.Accordion(get_text('misc'))
    with misc_accordion:
        with gr.Row():
            num_train_epochs = gr.Number(label=get_text('num_train_epochs'), value=default_config["num_train_epochs"], info=get_text('num_train_epochs_info'))
            save_model_epochs = gr.Number(label=get_text('save_model_epochs'), value=default_config["save_model_epochs"], info=get_text('save_model_epochs_info'))
            validation_epochs = gr.Number(label=get_text('validation_epochs'), value=default_config["validation_epochs"], info=get_text('validation_epochs_info'))
        with gr.Row():
            skip_epoch = gr.Number(label=get_text('skip_epoch'), value=default_config["skip_epoch"], info=get_text('skip_epoch_info'))
            skip_step = gr.Number(label=get_text('skip_step'), value=default_config["skip_step"], info=get_text('skip_step_info'))
            validation_ratio = gr.Number(label=get_text('validation_ratio'), value=default_config["validation_ratio"], info=get_text('validation_ratio_info'))
        with gr.Row():
            recreate_cache = gr.Checkbox(label=get_text('recreate_cache'), value=default_config["recreate_cache"])
            caption_dropout = gr.Number(label=get_text('caption_dropout'), value=default_config["caption_dropout"], info=get_text("caption_dropout_info"), maximum=1, minimum=0)
            max_time_steps = gr.Number(label=get_text('max_time_steps'), value=default_config["max_time_steps"], info=get_text("max_time_steps_info"), maximum=1000, minimum=0)
        # gr.Markdown(get_text('resolution_section'))
        resolution = gr.Dropdown(label=get_text('resolution'), value=default_config["resolution"], choices=default_config["resolution_choices"])

    slider_accordion = gr.Accordion(get_text('slider'))
    with slider_accordion:
        with gr.Row():
            use_two_captions = gr.Checkbox(label=get_text('use_two_captions'), value=default_config["use_two_captions"])
            slider_positive_scale = gr.Number(label=get_text('slider_positive_scale'), value=default_config["slider_positive_scale"])
            slider_negative_scale = gr.Number(label=get_text('slider_negative_scale'), value=default_config["slider_negative_scale"])

    config_template = gr.Accordion(get_text('config_template'))
    with config_template:
        with gr.Row():
            template_dropdown = gr.Dropdown(
                label=get_text('select_template'),
                choices=list_templates(default_config["config_path"]),
                value=os.path.basename(default_config["config_path"]),
                scale=2,
                allow_custom_value=True
            )
        with gr.Row():
            json_editor = gr.Code(language="json", label="Edit JSON Template", lines=25, interactive=True)
        with gr.Row():
            save_config_btn2 = gr.Button(get_text('save'), scale=1)
            load_config_btn2 = gr.Button(get_text('load'), scale=1)
    # save_edited_json_btn = gr.Button(get_text('save_edited_json'))

    output = gr.Textbox(label=get_text('output_box'))
    run_btn = gr.Button(get_text('run'))

    inputs = [
        config_path, script, seed, mixed_precision, report_to, lr_warmup_steps,
        output_dir, save_name, train_data_dir, optimizer, lr_scheduler, learning_rate,
        train_batch_size, repeats, gradient_accumulation_steps, num_train_epochs,
        save_model_epochs, validation_epochs, rank, skip_epoch, skip_step,
        gradient_checkpointing, validation_ratio, pretrained_model_name_or_path,
        model_path, resume_from_checkpoint, recreate_cache, resolution,
        caption_dropout, cosine_restarts, max_time_steps, blocks_to_swap,
        mask_dropout, reg_ratio, reg_timestep, use_two_captions,
        slider_positive_scale, slider_negative_scale,
        json_editor  # ← 就是这一项
    ]

    # Events
    run_btn.click(fn=run, inputs=inputs, outputs=output)
    save_config_btn.click(fn=save_config, inputs=inputs)
    load_config_btn.click(fn=load_config, inputs=[config_path], outputs=inputs)
    save_config_btn2.click(fn=save_config, inputs=inputs)
    load_config_btn2.click(fn=load_config, inputs=[config_path], outputs=inputs)
    
    script.change(fn=change_script, inputs=[script, config_path], outputs=[json_editor])
    
    template_dropdown.change(
        fn=load_template,
        inputs=[template_dropdown, config_path],
        outputs=[json_editor]
    )
    # save_edited_json_btn.click(
    #     fn=save_edited_json,
    #     inputs=[json_editor, config_path],
    #     outputs=[output]
    # )

    language_toggle_btn.click(
        fn=update_labels_only,
        inputs=[],
        outputs=[title_md, language_toggle_btn, template_dropdown, 
                #  save_edited_json_btn,
                script, config_path, save_config_btn, load_config_btn,
                output_dir, save_name, pretrained_model_name_or_path, resume_from_checkpoint,
                train_data_dir, model_path, report_to,
                rank, train_batch_size, repeats, gradient_accumulation_steps,
                mixed_precision, gradient_checkpointing, optimizer, lr_scheduler,
                cosine_restarts, learning_rate, lr_warmup_steps, seed,
                blocks_to_swap, mask_dropout, reg_ratio, reg_timestep,
                num_train_epochs, save_model_epochs, validation_epochs,
                skip_epoch, skip_step, validation_ratio, recreate_cache,
                caption_dropout, max_time_steps, resolution,
                use_two_captions, slider_positive_scale, slider_negative_scale,
                json_editor, output, run_btn,
                save_config_btn2, load_config_btn2  # ← 加上这两行
                ]
    )

    demo.load(
        fn=lambda: load_first_template(),
        outputs=[template_dropdown, template_dropdown, json_editor]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")