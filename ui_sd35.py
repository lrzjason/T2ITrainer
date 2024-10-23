import gradio as gr

import subprocess
import json
import sys
import os


default_config = {
    "script": "train_kolors_lora_ui.py",
    "script_choices": [
                        # "train_kolors_lora_ui.py",
                        "train_sd3_lora_ui.py"
                    #    "train_hunyuan_lora_ui.py","train_sd3_lora_ui.py"
                       ],
    "output_dir":"F:/models/sd3",
    "save_name":"sd35-lora",
    "pretrained_model_name_or_path":"F:/T2ITrainer/sd3.5L", # or local folder F:\Kolors
    "train_data_dir":"F:/ImageSet/3dkitten", 
    # "vae_path":None, # or local file
    "resume_from_checkpoint":None,
    # "model_path":None, 
    # "logging_dir":"logs",
    "report_to":"wandb", 
    "rank":4,
    "train_batch_size":1,
    "repeats":10,
    "gradient_accumulation_steps":1,
    "mixed_precision":"fp16",
    "gradient_checkpointing":True,
    "optimizer":"adamw",
    "lr_scheduler":"cosine", 
    "learning_rate":1e-4,
    "lr_warmup_steps":0,
    "seed":4321,
    "num_train_epochs":20,
    "save_model_epochs":1, 
    "validation_epochs":1, 
    "skip_epoch":0, 
    # "break_epoch":0,
    "skip_step":0, 
    "validation_ratio":0.1, 
    "use_dora":False,
    "recreate_cache":False,
    "caption_dropout":0.1,
    "config_path":"config.json",
    "resolution":"1024",
    "resolution_choices":["1024","512"],
    "use_debias":False,
    "snr_gamma":0,
    "cosine_restarts":1,
    "max_time_steps":0
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
        # model_path,
        resume_from_checkpoint,
        use_dora,
        recreate_cache,
        # vae_path,
        resolution,
        # use_debias,
        # snr_gamma,
        caption_dropout,
        cosine_restarts,
        max_time_steps
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
        # "model_path":model_path,
        "resume_from_checkpoint":resume_from_checkpoint,
        "use_dora":use_dora,
        "recreate_cache":recreate_cache,
        # "vae_path":vae_path,
        "config_path":config_path,
        "resolution":resolution,
        # "use_debias":use_debias,
        # 'snr_gamma':snr_gamma,
        "caption_dropout":caption_dropout,
        "cosine_restarts":cosine_restarts,
        "max_time_steps":max_time_steps
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
            default_config['pretrained_model_name_or_path'],default_config['resume_from_checkpoint'], \
            default_config['use_dora'],default_config['recreate_cache'],default_config['resolution'], \
            default_config['caption_dropout'], \
            default_config['cosine_restarts'],default_config['max_time_steps']
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
        # model_path,
        resume_from_checkpoint,
        use_dora,
        recreate_cache,
        # vae_path,
        resolution,
        # use_debias,
        # snr_gamma,
        caption_dropout,
        cosine_restarts,
        max_time_steps
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
        # "model_path":model_path,
        "resume_from_checkpoint":resume_from_checkpoint,
        "use_dora":use_dora,
        "recreate_cache":recreate_cache,
        # "vae_path":vae_path,
        "resolution":resolution,
        # "use_debias":use_debias,
        # "snr_gamma":snr_gamma,
        "caption_dropout":caption_dropout,
        "cosine_restarts":cosine_restarts,
        "max_time_steps":max_time_steps
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
        # model_path,
        resume_from_checkpoint,
        use_dora,
        recreate_cache,
        # vae_path,
        resolution,
        # use_debias,
        # snr_gamma,
        caption_dropout,
        cosine_restarts,
        max_time_steps
    )
    # print(args)
    return " ".join(args)
    

with gr.Blocks() as demo:
    gr.Markdown(
    """
    ## Lora Training
    """)
    script = gr.Dropdown(label="script", value=default_config["script"], choices=default_config["script_choices"])
    with gr.Row(equal_height=True):
            # Text input for user to specify another config save and load dir
        config_path = gr.Textbox(scale=3, label="Config Path (.json file)", value=default_config["config_path"], placeholder="Enter path to save/load config")
        save_config_btn = gr.Button("Save", scale=1)
        load_config_btn = gr.Button("load", scale=1)

    with gr.Accordion("Directory section"):
        # dir section
        with gr.Row():
            output_dir = gr.Textbox(label="output_dir", value=default_config["output_dir"],
                                   placeholder="checkpoint save to")
            save_name = gr.Textbox(label="save_name", value=default_config["save_name"],
                                   placeholder="checkpoint save name")
        with gr.Row():
            pretrained_model_name_or_path = gr.Textbox(label="pretrained_model_name_or_path", 
                value=default_config["pretrained_model_name_or_path"], 
                placeholder="repo name or dir contains diffusers model structure"
            )
            resume_from_checkpoint = gr.Textbox(label="resume_from_checkpoint", value=default_config["resume_from_checkpoint"], placeholder="resume the lora weight from seleted dir")
        with gr.Row():
            train_data_dir = gr.Textbox(label="train_data_dir", value=default_config["train_data_dir"], placeholder="dir contains dataset")
            report_to = gr.Dropdown(label="report_to", value=default_config["report_to"], choices=["wandb"])
            # logging_dir = gr.Textbox(label="logging_dir", value=default_config["logging_dir"], placeholder="logs folder")
        # with gr.Row():

    with gr.Accordion("Lora Config"):
        # train related section
        with gr.Row():
            rank = gr.Number(label="rank", value=default_config["rank"], info="Recommanded to use rank 4 for training set small than 100." )
            train_batch_size = gr.Number(label="train_batch_size", value=default_config["train_batch_size"], info="Batch size 1 is using 18GB. Please use small batch size to avoid oom." )
        with gr.Row():
            repeats = gr.Number(label="repeats", value=default_config["repeats"])
            gradient_accumulation_steps = gr.Number(label="gradient_accumulation_steps", value=default_config["gradient_accumulation_steps"])
            # mixed_precision = gr.Radio(label="mixed_precision", value=default_config["mixed_precision"], choices=["fp16", "bf16"])
            mixed_precision = gr.Radio(label="mixed_precision", value=default_config["mixed_precision"], choices=["fp16"])
            gradient_checkpointing = gr.Checkbox(label="gradient_checkpointing", value=default_config["gradient_checkpointing"])
            use_dora = gr.Checkbox(label="use_dora", value=default_config["use_dora"])
        with gr.Row():
            optimizer = gr.Dropdown(label="optimizer", value=default_config["optimizer"], choices=["adamw","prodigy"])
            lr_scheduler = gr.Dropdown(label="lr_scheduler", value=default_config["lr_scheduler"], 
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial","constant", "constant_with_warmup"])
            cosine_restarts = gr.Number(label="cosine_restarts", value=default_config["cosine_restarts"], info="Only useful for lr_scheduler: cosine_with_restarts", minimum=1)
        with gr.Row():
            learning_rate = gr.Number(label="learning_rate", value=default_config["learning_rate"], info="Recommended: 1e-4 or 1 for prodigy")
            lr_warmup_steps = gr.Number(label="lr_warmup_steps", value=default_config["lr_warmup_steps"])
            seed = gr.Number(label="seed", value=default_config["seed"])

    with gr.Accordion("Misc"):
        with gr.Row():
            num_train_epochs = gr.Number(label="num_train_epochs", value=default_config["num_train_epochs"], info="Total epoches of the training")
            save_model_epochs = gr.Number(label="save_model_epochs", value=default_config["save_model_epochs"], info="Save checkpoint when x epoches")
            validation_epochs = gr.Number(label="validation_epochs", value=default_config["validation_epochs"], info="perform validation when x epoches")
        with gr.Row():
            skip_epoch = gr.Number(label="skip_epoch", value=default_config["skip_epoch"], info="Skip x epoches for validation and save checkpoint")
            # break_epoch = gr.Number(label="break_epoch", value=default_config["break_epoch"], info="Stop train after x epoches")
            skip_step = gr.Number(label="skip_step", value=default_config["skip_step"], info="Skip x steps for validation and save checkpoint")
            validation_ratio = gr.Number(label="validation_ratio", value=default_config["validation_ratio"], info="Split dataset with this ratio for validation")
            
        with gr.Row():
            recreate_cache = gr.Checkbox(label="recreate_cache", value=default_config["recreate_cache"])
            # use_debias = gr.Checkbox(label="use_debias", value=default_config["use_debias"])
            # snr_gamma = gr.Number(label="min-snr_gamma recommanded: 5", value=default_config["snr_gamma"], info="Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.", maximum=10, minimum=0)
            caption_dropout = gr.Number(label="Caption Dropout", value=default_config["caption_dropout"], info="Caption Dropout", maximum=1, minimum=0)
            max_time_steps = gr.Number(label="Max timesteps limitation", value=default_config["max_time_steps"], info="Max timesteps limitation", maximum=1100, minimum=0)
        gr.Markdown(
"""
## Experiment Option: resolution
- Based target resolution (default:1024). 
- 512 or 1024 are supported.
""")
        with gr.Row():
            resolution = gr.Dropdown(label="resolution", value=default_config["resolution"], choices=default_config["resolution_choices"])
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
        # model_path,
        resume_from_checkpoint,
        use_dora,
        recreate_cache,
        # vae_path,
        resolution,
        # use_debias,
        # snr_gamma,
        caption_dropout,
        cosine_restarts,
        max_time_steps,
    ]
    output = gr.Textbox(label="Output Box")
    run_btn = gr.Button("Run")
    # inputs.append(config_path)
    run_btn.click(fn=run, inputs=inputs, outputs=output, api_name="run")
    save_config_btn.click(fn=save_config, inputs=inputs)
    load_config_btn.click(fn=load_config, inputs=[config_path], outputs=inputs)
demo.launch()