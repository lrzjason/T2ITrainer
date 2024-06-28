import gradio as gr

import subprocess
import json

default_config = {
    "default_script": "train_hunyuan_lora_ui.py",
    "script_choices": ["train_hunyuan_lora_ui.py","train_sd3_lora_ui.py"],
	"output_dir":"F:/models/hy",
    "save_name":"hy-lora",
    "pretrained_model_name_or_path":"Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers", 
    "model_path":None, 
    "train_data_dir":"F:/ImageSet/handpick_high_quality_b2_train", 
    "logging_dir":"logs",
    "report_to":"wandb", 
    "rank":32,
    "train_batch_size":1,
    "repeats":1,
    "gradient_accumulation_steps":1,
    "mixed_precision":"fp16",
    "gradient_checkpointing":True,
    "optimizer":"adamw",
    "lr_scheduler":"constant", 
    "learning_rate":1e-4,
    "lr_warmup_steps":0,
    "seed":4321,
    "num_train_epochs":20,
    "save_model_epochs":1, 
    "validation_epochs":1, 
    "skip_epoch":0, 
    "break_epoch":0,
    "skip_step":0, 
    "validation_ratio":0.1, 
    "use_dora":False,
    "recreate_cache":False,

}

def run(
        script,
        seed,
        logging_dir,
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
        break_epoch,
        skip_step,
        gradient_checkpointing,
        validation_ratio,
        pretrained_model_name_or_path,
        model_path,
        use_dora,
        recreate_cache
    ):
    inputs = {
        "seed":seed,
        "logging_dir":logging_dir,
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
        "break_epoch":break_epoch,
        "skip_step":skip_step,
        "gradient_checkpointing":gradient_checkpointing,
        "validation_ratio":validation_ratio,
        "pretrained_model_name_or_path":pretrained_model_name_or_path,
        # "model_path":model_path,
        "use_dora":use_dora,
        "recreate_cache":recreate_cache
    }
    # Convert the inputs dictionary to a list of arguments
    # args = ["python", "train_sd3_lora_ui.py"]  # replace "your_script.py" with the name of your script
    # script = "test_.pyt"
    args = ["python", script]
    for key, value in inputs.items():
        if value is not None:
            args.append(f"--{key}")
            if not isinstance(value, bool):  # exclude boolean values
                args.append(str(value))
    # Call the script with the arguments
    subprocess.run(args)
    print(args)
    return " ".join(args)
    

with gr.Blocks() as demo:
    script = gr.Dropdown(label="script", value=default_config["default_script"], choices=default_config["script_choices"])
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
                placeholder="repo name or dir contains sd3 medium diffusers structure"
            )
            model_path = gr.Textbox(visible=False, label="model_path", value=default_config["model_path"], placeholder="single weight files if not trained from official sd3 medium weight")
        with gr.Row():
            train_data_dir = gr.Textbox(label="train_data_dir", value=default_config["train_data_dir"], placeholder="dir contains dataset")
            logging_dir = gr.Textbox(label="logging_dir", value=default_config["logging_dir"], placeholder="logs folder")
        with gr.Row():
            report_to = gr.Dropdown(label="report_to", value=default_config["report_to"], choices=["wandb"])

    with gr.Accordion("Lora Config"):
        # train related section
        with gr.Row():
            rank = gr.Number(label="rank", value=default_config["rank"])
            train_batch_size = gr.Number(label="train_batch_size", value=default_config["train_batch_size"])
        with gr.Row():
            repeats = gr.Number(label="repeats", value=default_config["repeats"])
            gradient_accumulation_steps = gr.Number(label="gradient_accumulation_steps", value=default_config["gradient_accumulation_steps"])
            mixed_precision = gr.Radio(label="mixed_precision", value=default_config["mixed_precision"], choices=["fp16", "bf16"])
            gradient_checkpointing = gr.Checkbox(label="gradient_checkpointing", value=default_config["gradient_checkpointing"])
            use_dora = gr.Checkbox(label="use_dora", value=default_config["use_dora"])
        with gr.Row():
            optimizer = gr.Dropdown(label="optimizer", value=default_config["optimizer"], choices=["adamw","prodigy"])
            lr_scheduler = gr.Dropdown(label="lr_scheduler", value=default_config["lr_scheduler"], 
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial","constant", "constant_with_warmup"])
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
            break_epoch = gr.Number(label="break_epoch", value=default_config["break_epoch"], info="Stop train after x epoches")
            skip_step = gr.Number(label="skip_step", value=default_config["skip_step"], info="Skip x steps for validation and save checkpoint")
            validation_ratio = gr.Number(label="validation_ratio", value=default_config["validation_ratio"], info="Split dataset with this ratio for validation")
            recreate_cache = gr.Checkbox(label="recreate_cache", value=default_config["recreate_cache"])
    inputs = [
        script,
        seed,
        logging_dir,
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
        break_epoch,
        skip_step,
        gradient_checkpointing,
        validation_ratio,
        pretrained_model_name_or_path,
        model_path,
        use_dora,
        recreate_cache
        ]
    output = gr.Textbox(label="Output Box")
    run_btn = gr.Button("Run")
    run_btn.click(fn=run, inputs=inputs, outputs=output, api_name="run")
    
demo.launch()