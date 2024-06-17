import gradio as gr

import subprocess
import json

def run(
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
        model_path
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
        "model_path":model_path
    }
    # Convert the inputs dictionary to a list of arguments
    args = ["python", "train_sd3_lora_ui.py"]  # replace "your_script.py" with the name of your script
    for key, value in inputs.items():
        if value is not None:
            args.append(f"--{key}")
            if not isinstance(value, bool):  # exclude boolean values
                args.append(str(value))
    # Call the script with the arguments
    # subprocess.run(args)
    # print(args)
    return " ".join(args)
    

with gr.Blocks() as demo:
    with gr.Accordion("Directory section"):
        # dir section
        with gr.Row():
            output_dir = gr.Textbox(label="output_dir", value="F:/models/sd3",
                                   placeholder="checkpoint save to")
            save_name = gr.Textbox(label="save_name", value="opensd3",
                                   placeholder="checkpoint save name")
        with gr.Row():
            pretrained_model_name_or_path = gr.Textbox(label="pretrained_model_name_or_path", 
                value="stabilityai/stable-diffusion-3-medium-diffusers", 
                placeholder="repo name or dir contains sd3 medium diffusers structure"
            )
            model_path = gr.Textbox(label="model_path", value="F:/models/Stable-diffusion/sd3/opensd3_b2.safetensors", placeholder="single weight files if not trained from official sd3 medium weight")
        with gr.Row():
            train_data_dir = gr.Textbox(label="train_data_dir", value="F:/ImageSet/handpick_high_quality_b2_train", placeholder="dir contains dataset")
            logging_dir = gr.Textbox(label="logging_dir", value="logs", placeholder="logs folder")
        with gr.Row():
            report_to = gr.Dropdown(label="report_to", value="wandb", choices=["wandb"])

    with gr.Accordion("Lora Config"):
        # train related section
        with gr.Row():
            rank = gr.Number(label="rank", value=32)
            train_batch_size = gr.Number(label="train_batch_size", value=1)
        with gr.Row():
            repeats = gr.Number(label="repeats", value=10)
            gradient_accumulation_steps = gr.Number(label="gradient_accumulation_steps", value=10)
            mixed_precision = gr.Radio(label="mixed_precision", value="fp16", choices=["fp16", "bf16"])
            gradient_checkpointing = gr.Checkbox(label="gradient_checkpointing", value=True)
        with gr.Row():
            optimizer = gr.Dropdown(label="optimizer", value="adamw", choices=["adamw","prodigy"])
            lr_scheduler = gr.Dropdown(label="lr_scheduler", value="cosine", 
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial","constant", "constant_with_warmup"])
        with gr.Row():
            learning_rate = gr.Number(label="learning_rate", value=1e-4, info="Recommended: 1e-4 or 1 for prodigy")
            lr_warmup_steps = gr.Number(label="lr_warmup_steps", value=0)
            seed = gr.Number(label="seed", value=4321)

    with gr.Accordion("Misc"):
        with gr.Row():
            num_train_epochs = gr.Number(label="num_train_epochs", value=60, info="Total epoches of the training")
            save_model_epochs = gr.Number(label="save_model_epochs", value=1, info="Save checkpoint when x epoches")
            validation_epochs = gr.Number(label="validation_epochs", value=1, info="perform validation when x epoches")
        with gr.Row():
            skip_epoch = gr.Number(label="skip_epoch", value=1, info="Skip x epoches for validation and save checkpoint")
            break_epoch = gr.Number(label="break_epoch", value=0, info="Stop train after x epoches")
            skip_step = gr.Number(label="skip_step", value=0, info="Skip x steps for validation and save checkpoint")
            validation_ratio = gr.Number(label="validation_ratio", value=0.1, info="Split dataset with this ratio for validation")
        
    inputs = [
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
        model_path]
    output = gr.Textbox(label="Output Box")
    run_btn = gr.Button("Run")
    run_btn.click(fn=run, inputs=inputs, outputs=output, api_name="run")
    
demo.launch()