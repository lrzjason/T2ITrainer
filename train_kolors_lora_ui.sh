#!/bin/bash

# === required parameters ==
#Path to pretrained model or model identifier from huggingface.co/models.
pretrained_model_name_or_path=""

# The output directory where the model predictions and checkpoints will be written.
output_dir="output_dir"

# train data image folder
train_data_dir="train"

# seperate vae path
vae_path=""

# "default: '1024', accept str: '1024', '2048'"
resolution_config="1024"

# === optional parameters ===
# How many times to repeat the training data.
repeats=10

# Run validation every {x} epochs.
validation_epochs=1

# A seed for reproducible training.
seed=4321

# Batch size (per device) for the training dataloader.
train_batch_size=1

num_train_epochs=20

# Whether training should be resumed from a previous checkpoint. Use a path saved by
# `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.
resume_from_checkpoint=0

# save name prefix for saving checkpoints
save_name="kolors_lora_"

# Number of updates steps to accumulate before performing a backward/update pass.
gradient_accumulation_steps=1

# Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.
gradient_checkpointing=1

# Initial learning rate (after the potential warmup period) to use.
learning_rate=1e-4

# The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial","constant", "constant_with_warmup"]
lr_scheduler="cosine"

# Number of steps for the warmup in the lr scheduler.
lr_warmup_steps=0

# The optimizer type to use. Choose between ["AdamW", "prodigy"]
optimizer="adamw"

# [TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.
logging_dir="logs"

# The integration to report the results and logs to. Supported platforms are `"tensorboard"` (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.
report_to="wandb"

# Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >=
#  1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the
# flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config.
mixed_precision="fp16"

# The dimension of the LoRA update matrices.
rank=32

# Save model when {x} epochs
save_model_epochs=1

# skip val and save model before {x} epochs
skip_epoch=0

# skip val and save model before {x} step
skip_step=0

# break training after x epochs
break_epoch=0

# dataset split ratio for validation
validation_ratio=0.1

# seperate model path
model_path=""

# Use dora on peft config
use_dora=0

# recreate all cache
recreate_cache=1

# caption_dropout ratio which drop the caption and update the unconditional space
caption_dropout=0.1

# Use debiased estimation loss
use_debias=0

# SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0.  More details here: https://arxiv.org/abs/2303.09556.
snr_gamma=0

# check required parameters
if [ -z "$pretrained_model_name_or_path" ]; then
    echo "pretrained_model_name_or_path is required"
    exit 1
fi

if [ -z "$output_dir" ]; then
    echo "output_dir is required"
    exit 1
fi

if [ -z "$train_data_dir" ]; then
    echo "train_data_dir is required"
    exit 1
fi

if [ -z "$vae_path" ]; then
    echo "vae_path is required"
    exit 1
fi


# define dynamic parameters based on the resolution
if [ $resume_from_checkpoint -eq 1 ]; then
    extra_args="--resume_from_checkpoint"
fi

if [ $gradient_checkpointing -eq 1 ]; then
    extra_args="$extra_args --gradient_checkpointing"
fi

if [ $use_dora -eq 1 ]; then
    extra_args="$extra_args --use_dora"
fi

if [ $recreate_cache -eq 1 ]; then
    extra_args="$extra_args --recreate_cache"
fi

if [ $use_debias -eq 1 ]; then
    extra_args="$extra_args --use_debias"
fi

python train_kolors_lora_ui.py \
    --pretrained_model_name_or_path $pretrained_model_name_or_path \
    --output_dir $output_dir \
    --train_data_dir $train_data_dir \
    --vae_path $vae_path \
    --resolution_config $resolution_config \
    --repeats $repeats \
    --validation_epochs $validation_epochs \
    --seed $seed \
    --train_batch_size $train_batch_size \
    --num_train_epochs $num_train_epochs \
    --save_name $save_name \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --learning_rate $learning_rate \
    --lr_scheduler $lr_scheduler \
    --lr_warmup_steps $lr_warmup_steps \
    --optimizer $optimizer \
    --logging_dir $logging_dir \
    --report_to $report_to \
    --mixed_precision $mixed_precision \
    --rank $rank \
    --save_model_epochs $save_model_epochs \
    --skip_epoch $skip_epoch \
    --skip_step $skip_step \
    --break_epoch $break_epoch \
    --validation_ratio $validation_ratio \
    --model_path $model_path \
    --caption_dropout $caption_dropout \
    --snr_gamma $snr_gamma \
    $extra_args
