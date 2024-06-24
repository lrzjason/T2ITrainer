default_config = {
	"output_dir":"F:/models/sd3",
    "save_name":"opensd3",
    "pretrained_model_name_or_path":"stabilityai/stable-diffusion-3-medium-diffusers", 
    "model_path":"F:/models/Stable-diffusion/sd3/opensd3_b2.safetensors", 
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
    "skip_epoch":1, 
    "break_epoch":0,
    "skip_step":0, 
    "validation_ratio":0.1, 
}
def method_a(**kwargs):
    print(kwargs)
    
method_a(default_config=default_config)