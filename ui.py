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
    subprocess.run(args)
    # print(args)
    return " ".join(args)
    
inputs = [
    gr.Number(label="seed", value=4321),
    gr.Textbox(label="logging_dir", value="logs"),
    gr.Textbox(label="mixed_precision", value="fp16"),
    gr.Textbox(label="report_to", value="wandb"),
    gr.Number(label="lr_warmup_steps", value=1),
    gr.Textbox(label="output_dir", value="F:/models/sd3"),
    gr.Textbox(label="save_name", value="opensd3"),
    gr.Textbox(label="train_data_dir", value="F:/ImageSet/handpick_high_quality_b2_train"),
    gr.Textbox(label="optimizer", value="adamw"),
    gr.Textbox(label="lr_scheduler", value="cosine"),
    gr.Number(label="learning_rate", value=1e-4),
    gr.Number(label="train_batch_size", value=1),
    gr.Number(label="repeats", value=10),
    gr.Number(label="gradient_accumulation_steps", value=10),
    gr.Number(label="num_train_epochs", value=60),
    gr.Number(label="save_model_epochs", value=1),
    gr.Number(label="validation_epochs", value=1),
    gr.Number(label="rank", value=32),
    gr.Number(label="skip_epoch", value=1),
    gr.Number(label="break_epoch", value=0),
    gr.Number(label="skip_step", value=0),
    gr.Radio(label="gradient_checkpointing", value=True),
    gr.Number(label="validation_ratio", value=0.1),
    gr.Textbox(label="pretrained_model_name_or_path", value="stabilityai/stable-diffusion-3-medium-diffusers"),
    gr.Textbox(label="model_path", value="F:/models/Stable-diffusion/sd3/opensd3_b2.safetensors"),
]

outputs = [
    gr.Textbox(label="output"),
]

with gr.Blocks() as demo:
    seed = gr.Number(label="seed", value=4321)
    logging_dir = gr.Textbox(label="logging_dir", value="logs")
    mixed_precision = gr.Textbox(label="mixed_precision", value="fp16")
    report_to = gr.Textbox(label="report_to", value="wandb")
    lr_warmup_steps = gr.Number(label="lr_warmup_steps", value=1)
    output_dir = gr.Textbox(label="output_dir", value="F:/models/sd3")
    save_name = gr.Textbox(label="save_name", value="opensd3")
    train_data_dir = gr.Textbox(label="train_data_dir", value="F:/ImageSet/handpick_high_quality_b2_train")
    optimizer = gr.Textbox(label="optimizer", value="adamw")
    lr_scheduler = gr.Textbox(label="lr_scheduler", value="cosine")
    learning_rate = gr.Number(label="learning_rate", value=1e-4)
    train_batch_size = gr.Number(label="train_batch_size", value=1)
    repeats = gr.Number(label="repeats", value=10)
    gradient_accumulation_steps = gr.Number(label="gradient_accumulation_steps", value=10)
    num_train_epochs = gr.Number(label="num_train_epochs", value=60)
    save_model_epochs = gr.Number(label="save_model_epochs", value=1)
    validation_epochs = gr.Number(label="validation_epochs", value=1)
    rank = gr.Number(label="rank", value=32)
    skip_epoch = gr.Number(label="skip_epoch", value=1)
    break_epoch = gr.Number(label="break_epoch", value=0)
    skip_step = gr.Number(label="skip_step", value=0)
    gradient_checkpointing = gr.Radio(label="gradient_checkpointing", value=True)
    validation_ratio = gr.Number(label="validation_ratio", value=0.1)
    pretrained_model_name_or_path = gr.Textbox(label="pretrained_model_name_or_path", value="stabilityai/stable-diffusion-3-medium-diffusers")
    model_path = gr.Textbox(label="model_path", value="F:/models/Stable-diffusion/sd3/opensd3_b2.safetensors")
    
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