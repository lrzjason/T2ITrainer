import gradio as gr

import subprocess

def process_args(inputs):
    # Convert the inputs dictionary to a list of arguments
    args = ["python", "test.py"]  # replace "your_script.py" with the name of your script
    for key, value in inputs.items():
        if value is not None:
            args.append(f"--{key}")
            if not isinstance(value, bool):  # exclude boolean values
                args.append(str(value))
    # Call the script with the arguments
    subprocess.run(args)
    
inputs = [
    gr.inputs.Textbox(label="seed", type="number", default=4321),
    gr.inputs.Textbox(label="logging_dir", default="logs"),
    gr.inputs.Textbox(label="mixed_precision", default="fp16"),
    gr.inputs.Textbox(label="report_to", default="wandb"),
    gr.inputs.Textbox(label="lr_warmup_steps", type="number", default=1),
    gr.inputs.Textbox(label="resolution", type="number", default=1024),
    gr.inputs.Textbox(label="output_dir", default="F:/models/sd3"),
    gr.inputs.Textbox(label="save_name", default="opensd3"),
    gr.inputs.Textbox(label="train_data_dir", default="F:/ImageSet/handpick_high_quality_b2_train"),
    gr.inputs.Textbox(label="optimizer", default="adamw"),
    gr.inputs.Textbox(label="lr_scheduler", default="cosine"),
    gr.inputs.Textbox(label="learning_rate", type="number", default=1e-4),
    gr.inputs.Textbox(label="train_batch_size", type="number", default=1),
    gr.inputs.Textbox(label="repeats", type="number", default=10),
    gr.inputs.Textbox(label="gradient_accumulation_steps", type="number", default=10),
    gr.inputs.Textbox(label="num_train_epochs", type="number", default=60),
    gr.inputs.Textbox(label="save_model_epochs", type="number", default=1),
    gr.inputs.Textbox(label="validation_epochs", type="number", default=1),
    gr.inputs.Textbox(label="rank", type="number", default=32),
    gr.inputs.Textbox(label="skip_epoch", type="number", default=1),
    gr.inputs.Textbox(label="break_epoch", type="number", default=0),
    gr.inputs.Textbox(label="skip_step", type="number", default=0),
    gr.inputs.Textbox(label="gradient_checkpointing", type="boolean", default=True),
    gr.inputs.Textbox(label="validation_ratio", type="number", default=0.1),
    gr.inputs.Textbox(label="pretrained_model_name_or_path", default="stabilityai/stable-diffusion-3-medium-diffusers"),
    gr.inputs.Textbox(label="model_path", default="F:/models/Stable-diffusion/sd3/opensd3_b2.safetensors"),
    , outputs=
]

iface = gr.Interface(fn=process_args, inputs=inputs, outputs=)
iface.launch()