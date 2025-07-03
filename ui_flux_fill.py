import gradio as gr

import subprocess
import json
import sys
import os


default_config = {
    "script": "train_flux_lora_ui_kontext.py",
    "script_choices": [
                        "train_flux_lora_ui_kontext.py",
                        "train_flux_lora_ui_with_mask.py",
                        "train_flux_lora_ui.py",
                       ],
    "output_dir":"F:/models/flux",
    "save_name":"flux-lora",
    "pretrained_model_name_or_path":"F:/T2ITrainer/flux_models/kontext", # or local folder F:\Kolors
    "train_data_dir":"F:/ImageSet/kontext", 
    # "vae_path":None, # or local file
    "resume_from_checkpoint":None,
    "model_path":None, 
    # "logging_dir":"logs",
    "report_to":"wandb", 
    "rank":16,
    "train_batch_size":1,
    "repeats":1,
    "gradient_accumulation_steps":1,
    "mixed_precision":"bf16",
    "gradient_checkpointing":True,
    "optimizer":"adamw",
    "lr_scheduler":"constant", 
    "learning_rate":1e-4,
    "lr_warmup_steps":0,
    "seed":4321,
    "num_train_epochs":5,
    "save_model_epochs":1, 
    "validation_epochs":1, 
    "skip_epoch":0, 
    # "break_epoch":0,
    "skip_step":0, 
    "validation_ratio":0.1, 
    # "use_dora":False,
    "recreate_cache":False,
    "caption_dropout":0.1,
    "config_path":"config.json",
    "resolution":"512",
    "resolution_choices":["512","256","1024"],
    "use_debias":False,
    "snr_gamma":0,
    "cosine_restarts":1,
    "max_time_steps":0,
    "blocks_to_swap":0,
    "mask_dropout":0,
    "reg_ratio":0.7,
    "reg_timestep":700
    # "use_fp8":True
    # "freeze_transformer_layers":'5,7,10,17,18,19'
}


# Function to save configuration to a specified directory
def save_config( 
        config_path,
        script,
        seed,
        # logging_dir,
        mixed_precision,
        report_to,
        lr_warmup_steps,
        output_dir,
        save_name,
        train_data_dir,
        optimizer,
        lr_scheduler,
        learning_rate,
        train_batch_size,
        repeats,
        gradient_accumulation_steps,
        num_train_epochs,
        save_model_epochs,
        validation_epochs,
        rank,
        skip_epoch,
        # break_epoch,
        skip_step,
        gradient_checkpointing,
        validation_ratio,
        pretrained_model_name_or_path,
        model_path,
        resume_from_checkpoint,
        # use_dora,
        recreate_cache,
        # vae_path,
        resolution,
        # use_debias,
        # snr_gamma,
        caption_dropout,
        cosine_restarts,
        max_time_steps,
        blocks_to_swap,
        mask_dropout,
        reg_ratio,
        reg_timestep
        # use_fp8
        # freeze_transformer_layers
    ):
    config = {
        "script":script,
        "seed":seed,
        # "logging_dir":logging_dir,
        "mixed_precision":mixed_precision,
        "report_to":report_to,
        "lr_warmup_steps":lr_warmup_steps,
        "output_dir":output_dir,
        "save_name":save_name,
        "train_data_dir":train_data_dir,
        "optimizer":optimizer,
        "lr_scheduler":lr_scheduler,
        "learning_rate":learning_rate,
        "train_batch_size":train_batch_size,
        "repeats":repeats,
        "gradient_accumulation_steps":gradient_accumulation_steps,
        "num_train_epochs":num_train_epochs,
        "save_model_epochs":save_model_epochs,
        "validation_epochs":validation_epochs,
        "rank":rank,
        "skip_epoch":skip_epoch,
        # "break_epoch":break_epoch,
        "skip_step":skip_step,
        "gradient_checkpointing":gradient_checkpointing,
        "validation_ratio":validation_ratio,
        "pretrained_model_name_or_path":pretrained_model_name_or_path,
        "model_path":model_path,
        "resume_from_checkpoint":resume_from_checkpoint,
        # "use_dora":use_dora,
        "recreate_cache":recreate_cache,
        # "vae_path":vae_path,
        "config_path":config_path,
        "resolution":resolution,
        # "use_debias":use_debias,
        # 'snr_gamma':snr_gamma,
        "caption_dropout":caption_dropout,
        "cosine_restarts":cosine_restarts,
        "max_time_steps":max_time_steps,
        # "freeze_transformer_layers":freeze_transformer_layers
        "blocks_to_swap":blocks_to_swap,
        "mask_dropout":mask_dropout,
        "reg_ratio":reg_ratio,
        "reg_timestep":reg_timestep
        # "use_fp8":use_fp8
    }
    # config_path = os.path.join(config_dir, f"{filename}{ext}")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {config_path}")
    print(f"Update default config")
    with open("config.json", 'w') as f:
        json.dump(config, f, indent=4)

# Function to load configuration from a specified directory
def load_config(config_path):
    if not config_path.endswith(".json"):
        print("!!!File is not json format.")
        print("Load default config")
        config_path = "config.json"
    if not os.path.exists(config_path):
        # create config
        with open(config_path, 'w') as f:
            config = {}
            for key in default_config.keys():
                config[key] = default_config[key]
            json.dump(config, f, indent=4)
        return config
    config_path = os.path.join(config_path)
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except:
        config_path = "config.json"
    print(f"Loaded configuration from {config_path}")
    # print("config")
    # print(config)
    for key in config.keys():
        default_config[key] = config[key]
            
    # print("default_config")
    # print(default_config)
    return config_path,default_config['script'],default_config['seed'], \
            default_config['mixed_precision'],default_config['report_to'],default_config['lr_warmup_steps'], \
            default_config['output_dir'],default_config['save_name'],default_config['train_data_dir'], \
            default_config['optimizer'],default_config['lr_scheduler'],default_config['learning_rate'], \
            default_config['train_batch_size'],default_config['repeats'],default_config['gradient_accumulation_steps'], \
            default_config['num_train_epochs'],default_config['save_model_epochs'],default_config['validation_epochs'], \
            default_config['rank'],default_config['skip_epoch'], \
            default_config['skip_step'],default_config['gradient_checkpointing'],default_config['validation_ratio'], \
            default_config['pretrained_model_name_or_path'],default_config['model_path'],default_config['resume_from_checkpoint'], \
            default_config['recreate_cache'],default_config['resolution'], \
            default_config['caption_dropout'], \
            default_config['cosine_restarts'],default_config['max_time_steps'], \
            default_config['blocks_to_swap'],default_config['mask_dropout'], \
            default_config['reg_ratio'],default_config['reg_timestep']
            # default_config['use_dora'], \
            # default_config['freeze_transformer_layers']
            # default_config['logging_dir'],default_config['break_epoch'], 

# load config.json by default
load_config("config.json")
def run(
        config_path,
        script,
        seed,
        # logging_dir,
        mixed_precision,
        report_to,
        lr_warmup_steps,
        output_dir,
        save_name,
        train_data_dir,
        optimizer,
        lr_scheduler,
        learning_rate,
        train_batch_size,
        repeats,
        gradient_accumulation_steps,
        num_train_epochs,
        save_model_epochs,
        validation_epochs,
        rank,
        skip_epoch,
        # break_epoch,
        skip_step,
        gradient_checkpointing,
        validation_ratio,
        pretrained_model_name_or_path,
        model_path,
        resume_from_checkpoint,
        # use_dora,
        recreate_cache,
        # vae_path,
        resolution,
        # use_debias,
        # snr_gamma,
        caption_dropout,
        cosine_restarts,
        max_time_steps,
        blocks_to_swap,
        mask_dropout,
        reg_ratio,
        reg_timestep
        # use_fp8
        # freeze_transformer_layers
    ):
    # if vae_path is not None:
    #     if not vae_path.endswith('.safetensors') and not vae_path == "":
    #         msg = "Vae need to be a single file ends with .safetensors. It should be the fp16 fix vae from https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/tree/main"
    #         gr.Warning(msg)
    #         return msg
    
    inputs = {
        "seed":seed,
        # "logging_dir":logging_dir,
        "mixed_precision":mixed_precision,
        "report_to":report_to,
        "lr_warmup_steps":lr_warmup_steps,
        "output_dir":output_dir,
        "save_name":save_name,
        "train_data_dir":train_data_dir,
        "optimizer":optimizer,
        "lr_scheduler":lr_scheduler,
        "learning_rate":learning_rate,
        "train_batch_size":train_batch_size,
        "repeats":repeats,
        "gradient_accumulation_steps":gradient_accumulation_steps,
        "num_train_epochs":num_train_epochs,
        "save_model_epochs":save_model_epochs,
        "validation_epochs":validation_epochs,
        "rank":rank,
        "skip_epoch":skip_epoch,
        # "break_epoch":break_epoch,
        "skip_step":skip_step,
        "gradient_checkpointing":gradient_checkpointing,
        "validation_ratio":validation_ratio,
        "pretrained_model_name_or_path":pretrained_model_name_or_path,
        "model_path":model_path,
        "resume_from_checkpoint":resume_from_checkpoint,
        # "use_dora":use_dora,
        "recreate_cache":recreate_cache,
        # "vae_path":vae_path,
        "resolution":resolution,
        # "use_debias":use_debias,
        # "snr_gamma":snr_gamma,
        "caption_dropout":caption_dropout,
        "cosine_restarts":cosine_restarts,
        "max_time_steps":max_time_steps,
        "blocks_to_swap":blocks_to_swap,
        "mask_dropout":mask_dropout,
        "reg_ratio":reg_ratio,
        "reg_timestep":reg_timestep
        # "use_fp8":use_fp8
        # "freeze_transformer_layers":freeze_transformer_layers
    }
    # Convert the inputs dictionary to a list of arguments
    # args = ["python", "train_sd3_lora_ui.py"]  # replace "your_script.py" with the name of your script
    # script = "test_.pyt"
    args = [sys.executable, script]
    for key, value in inputs.items():
        if value is not None:
            if isinstance(value, bool):  # exclude boolean values
                if value == True:
                    args.append(f"--{key}")
            else:
                args.append(f"--{key}")
                args.append(str(value))
    
    # Add the config_path argument if the script is train_flux_lora_ui_with_mask_cat_custom3.py
    # if script == "train_flux_lora_ui_with_mask_cat_custom3.py":
    #     args.append("--config_path")
    #     args.append(config_path)
    
    # Call the script with the arguments
    subprocess.call(args)
    save_config(
        config_path,
        script,
        seed,
        # logging_dir,
        mixed_precision,
        report_to,
        lr_warmup_steps,
        output_dir,
        save_name,
        train_data_dir,
        optimizer,
        lr_scheduler,
        learning_rate,
        train_batch_size,
        repeats,
        gradient_accumulation_steps,
        num_train_epochs,
        save_model_epochs,
        validation_epochs,
        rank,
        skip_epoch,
        # break_epoch,
        skip_step,
        gradient_checkpointing,
        validation_ratio,
        pretrained_model_name_or_path,
        model_path,
        resume_from_checkpoint,
        # use_dora,
        recreate_cache,
        # vae_path,
        resolution,
        # use_debias,
        # snr_gamma,
        caption_dropout,
        cosine_restarts,
        max_time_steps,
        blocks_to_swap,
        mask_dropout,
        reg_ratio,
        reg_timestep
        # use_fp8
        # freeze_transformer_layers
    )
    # print(args)
    return " ".join(args)
    

with gr.Blocks(title="Flux Kontext LoRA 训练界面", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
    """
    # 🎨 Flux Kontext LoRA 训练界面
    专为Flux Kontext模型LoRA训练设计的图形界面
    """)
    script = gr.Dropdown(label="🔧 训练脚本", value=default_config["script"], choices=default_config["script_choices"], info="选择要使用的训练脚本")
    with gr.Row(equal_height=True):
            # Text input for user to specify another config save and load dir
        config_path = gr.Textbox(scale=3, label="📁 配置文件路径 (.json文件)", value=default_config["config_path"], placeholder="输入保存/加载配置的路径")
        save_config_btn = gr.Button("💾 保存配置", scale=1, variant="secondary")
        load_config_btn = gr.Button("📂 加载配置", scale=1, variant="secondary")

    with gr.Accordion("📂 目录配置", open=True):
        # dir section
        with gr.Row():
            output_dir = gr.Textbox(label="📤 输出目录", value=default_config["output_dir"],
                                   placeholder="模型保存位置")
            save_name = gr.Textbox(label="💾 保存名称", value=default_config["save_name"],
                                   placeholder="模型保存名称")
        with gr.Row():
            pretrained_model_name_or_path = gr.Textbox(label="🤖 预训练模型路径", 
                value=default_config["pretrained_model_name_or_path"], 
                placeholder="模型仓库名称或包含diffusers模型结构的目录"
            )
            resume_from_checkpoint = gr.Textbox(label="🔄 从检查点恢复", value=default_config["resume_from_checkpoint"], placeholder="从选定目录恢复LoRA权重")
        with gr.Row():
            train_data_dir = gr.Textbox(label="📊 训练数据目录", value=default_config["train_data_dir"], placeholder="包含数据集的目录")
            model_path = gr.Textbox(label="📄 模型文件路径", value=default_config["model_path"], placeholder="单个权重文件路径（如果不是从官方权重训练）")
            # logging_dir = gr.Textbox(label="logging_dir", value=default_config["logging_dir"], placeholder="logs folder")
        with gr.Row():
            report_to = gr.Dropdown(label="📈 报告平台", value=default_config["report_to"], choices=["wandb"], info="训练指标报告平台")
            

    with gr.Accordion("⚙️ LoRA 配置", open=True):
        # train related section
        with gr.Row():
            rank = gr.Number(label="🔢 LoRA Rank", value=default_config["rank"], info="建议：训练集少于100时使用rank 4，一般用16-32" )
            train_batch_size = gr.Number(label="📦 训练批次大小", value=default_config["train_batch_size"], info="批次大小1使用18GB显存，请使用小批次避免显存不足" )
        with gr.Row():
            repeats = gr.Number(label="🔄 重复次数", value=default_config["repeats"], info="每个样本重复训练次数")
            gradient_accumulation_steps = gr.Number(label="📈 梯度累积步数", value=default_config["gradient_accumulation_steps"], info="累积梯度减少显存使用")
            # mixed_precision = gr.Radio(label="mixed_precision", value=default_config["mixed_precision"], choices=["fp16", "bf16"])
            mixed_precision = gr.Radio(label="💾 混合精度", value=default_config["mixed_precision"], choices=["bf16", "fp8"], info="推荐bf16")
            gradient_checkpointing = gr.Checkbox(label="🔍 梯度检查点", value=default_config["gradient_checkpointing"], info="节省显存但稍慢")
            # use_dora = gr.Checkbox(label="use_dora", value=default_config["use_dora"])
        with gr.Row():
            optimizer = gr.Dropdown(label="🎯 优化器", value=default_config["optimizer"], choices=["adamw","prodigy"], info="推荐AdamW")
            lr_scheduler = gr.Dropdown(label="📊 学习率调度器", value=default_config["lr_scheduler"], 
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial","constant", "constant_with_warmup"], info="推荐cosine")
            cosine_restarts = gr.Number(label="🔄 余弦重启次数", value=default_config["cosine_restarts"], info="仅对cosine_with_restarts有效", minimum=1)
        with gr.Row():
            learning_rate = gr.Number(label="📈 学习率", value=default_config["learning_rate"], info="推荐：1e-4（AdamW）或1（Prodigy）")
            lr_warmup_steps = gr.Number(label="🔥 学习率预热步数", value=default_config["lr_warmup_steps"], info="预热步数")
            seed = gr.Number(label="🎲 随机种子", value=default_config["seed"], info="固定随机种子保证可重现性")
        with gr.Row():
            blocks_to_swap = gr.Number(label="💾 块交换数量", value=default_config["blocks_to_swap"], info="交换到CPU的块数量。24GB显存建议10，更低显存用更多" )
            mask_dropout = gr.Number(label="🎭 掩码丢弃率", value=default_config["mask_dropout"], info="掩码丢弃意味着掩码全为1进行整图重建" )
        #     freeze_transformer_layers = gr.Textbox(label="freeze_transformer_layers", value=default_config["freeze_transformer_layers"], info="Stop training the transformer layers included in the input using ',' to seperate layers. Example: 5,7,10,17,18,19" )
            reg_ratio = gr.Number(label="📊 正则化比例", value=default_config["reg_ratio"], info="目标迁移学习的正则化。如果不训练不同目标设为1" )
            reg_timestep = gr.Number(label="⏰ 正则化时间步", value=default_config["reg_timestep"], info="目标迁移学习的正则化。如果不训练不同目标设为0" )
            
            
    with gr.Accordion("🔧 其他设置"):
        with gr.Row():
            num_train_epochs = gr.Number(label="🔄 训练轮数", value=default_config["num_train_epochs"], info="训练的总轮数")
            save_model_epochs = gr.Number(label="💾 保存模型间隔", value=default_config["save_model_epochs"], info="每x轮保存检查点")
            validation_epochs = gr.Number(label="🔍 验证间隔", value=default_config["validation_epochs"], info="每x轮执行验证")
        with gr.Row():
            skip_epoch = gr.Number(label="⏭️ 跳过轮数", value=default_config["skip_epoch"], info="跳过x轮进行验证和保存检查点")
            # break_epoch = gr.Number(label="break_epoch", value=default_config["break_epoch"], info="Stop train after x epoches")
            skip_step = gr.Number(label="⏭️ 跳过步数", value=default_config["skip_step"], info="跳过x步进行验证和保存检查点")
            validation_ratio = gr.Number(label="📊 验证集比例", value=default_config["validation_ratio"], info="按此比例分割数据集用于验证")
            
        with gr.Row():
            recreate_cache = gr.Checkbox(label="🔄 重新创建缓存", value=default_config["recreate_cache"], info="强制重新生成所有缓存文件")
            # use_debias = gr.Checkbox(label="use_debias", value=default_config["use_debias"])
            # snr_gamma = gr.Number(label="min-snr_gamma recommanded: 5", value=default_config["snr_gamma"], info="Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.", maximum=10, minimum=0)
            caption_dropout = gr.Number(label="💬 标题丢弃率", value=default_config["caption_dropout"], info="随机丢弃标题的概率", maximum=1, minimum=0)
            max_time_steps = gr.Number(label="⏰ 最大时间步限制", value=default_config["max_time_steps"], info="最大时间步数限制", maximum=1000, minimum=0)
        gr.Markdown(
"""
## 🎯 实验选项：分辨率
- 基于目标分辨率（默认：1024）
- 支持512或1024分辨率
- **推荐**：显存不足时使用512，充足时使用1024
""")
        with gr.Row():
            resolution = gr.Dropdown(label="🖼️ 训练分辨率", value=default_config["resolution"], choices=default_config["resolution_choices"], info="更高分辨率需要更多显存")
    inputs = [
        config_path,
        script,
        seed,
        # logging_dir,
        mixed_precision,
        report_to,
        lr_warmup_steps,
        output_dir,
        save_name,
        train_data_dir,
        optimizer,
        lr_scheduler,
        learning_rate,
        train_batch_size,
        repeats,
        gradient_accumulation_steps,
        num_train_epochs,
        save_model_epochs,
        validation_epochs,
        rank,
        skip_epoch,
        # break_epoch,
        skip_step,
        gradient_checkpointing,
        validation_ratio,
        pretrained_model_name_or_path,
        model_path,
        resume_from_checkpoint,
        # use_dora,
        recreate_cache,
        # vae_path,
        resolution,
        # use_debias,
        # snr_gamma,
        caption_dropout,
        cosine_restarts,
        max_time_steps,
        blocks_to_swap,
        mask_dropout,
        reg_ratio,
        reg_timestep
        # freeze_transformer_layers,
    ]
    # 添加训练提示信息
    gr.Markdown(
    """
    ---
    ### 💡 训练提示
    - **显存不足**：减小批次大小、增加块交换数量、降低分辨率
    - **推荐设置**：Rank 16-32，学习率1e-4，AdamW优化器，bf16精度
    - **数据准备**：确保训练数据目录包含配对的_R(参考)和_T(目标)图像及对应的.txt文件
    """
    )
    
    output = gr.Textbox(label="📄 训练输出日志", lines=10, max_lines=20, show_copy_button=True, interactive=False)
    
    with gr.Row():
        run_btn = gr.Button("🚀 开始训练", variant="primary", size="lg")
        
    # 状态显示
    gr.Markdown(
    """
    ---
    ### 📊 当前配置状态
    请确认以上设置后点击"开始训练"按钮。训练过程中的详细日志将在上方输出框中显示。
    """
    )
    # inputs.append(config_path)
    run_btn.click(fn=run, inputs=inputs, outputs=output, api_name="run")
    save_config_btn.click(fn=save_config, inputs=inputs)
    load_config_btn.click(fn=load_config, inputs=[config_path], outputs=inputs)
demo.launch()