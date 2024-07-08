import gradio as gr

import subprocess
import json

default_config = {
    "script": "merge_state_dict.py",
	"kolors_model_path":"/path/to/kolors/unet/diffusion_pytorch_model.fp16.safetensors",
    "sdxl_model_path":"/path/to/yoursdxlweight.safetensors",
    "merged_kolors_path":"/path/to/kolors/unet/merged_diffusion_pytorch_model.fp16.safetensors",
    "ratio":0.25,
    "perturbed_ratio":0
}

def run(
        script,
        kolors_model_path,
        sdxl_model_path,
        merged_kolors_path,
        ratio,
        perturbed_ratio
    ):
    inputs = {
        "kolors_model_path":kolors_model_path,
        "sdxl_model_path":sdxl_model_path,
        "merged_kolors_path":merged_kolors_path,
        "ratio":ratio,
        "perturbed_ratio":perturbed_ratio
    }
    # Convert the inputs dictionary to a list of arguments
    # args = ["python", "train_sd3_lora_ui.py"]  # replace "your_script.py" with the name of your script
    # script = "test_.pyt"
    args = ["python", script]
    for key, value in inputs.items():
        if value is not None:
            if isinstance(value, bool):  # exclude boolean values
                if value == True:
                    args.append(f"--{key}")
            else:
                args.append(f"--{key}")
                args.append(str(value))
                
    # Call the script with the arguments
    print(args)
    subprocess.run(args)
    return " ".join(args)
    

with gr.Blocks() as demo:
    script = gr.Textbox(label="script", value=default_config["script"])
    with gr.Accordion("Directory section"):
        # dir section
        with gr.Row():
            kolors_model_path = gr.Textbox(label="Kolors Model Path", value=default_config["kolors_model_path"],
                                   placeholder="Kolors unet .safetensors file. Example: F:/models/Kolors/unet/diffusion_pytorch_model.fp16.safetensors")
            sdxl_model_path = gr.Textbox(label="SDXL Model Path", value=default_config["sdxl_model_path"],
                                   placeholder="SDXL .safetensors file. Example: F:/models/Stable-diffusion/yoursdxlmodel.safetensors")
        with gr.Row():
            merged_kolors_path = gr.Textbox(label="Merged Kolors Path", value=default_config["merged_kolors_path"],
                                   placeholder="Merged Kolors unet .safetensors file. Example: F:/models/Kolors/unet/merged_diffusion_pytorch_model.fp16.safetensors")

    with gr.Accordion("Merge Config"):
        # train related section
        gr.Markdown("Merging Formula: Kolors weight * (1 - ratio) + SDXL weight * ratio. Merge too much ratio would collapse the model. Recommanded in 0.1~0.5. \n In a few tests, more than 0.25 ratio would affect Chinese character accuracy.")
        with gr.Row():
            ratio = gr.Number(label="ratio", value=default_config["ratio"], minimum=0, maximum=1, step=0.01)
        gr.Markdown("Experiment Function. Add some randomness to the merged. Default: 0. If you want to try it, recommanded 0.02 no more than 0.03")
        with gr.Row():
            perturbed_ratio = gr.Number(label="perturbed_ratio", value=default_config["perturbed_ratio"], minimum=0, maximum=0.1, step=0.001)
        
    inputs = [
        script,
        kolors_model_path,
        sdxl_model_path,
        merged_kolors_path,
        ratio,
        perturbed_ratio
        ]
    output = gr.Textbox(label="Output Box")
    run_btn = gr.Button("Run")
    run_btn.click(fn=run, inputs=inputs, outputs=output, api_name="run")
    
demo.launch()