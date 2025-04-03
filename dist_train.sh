#!/bin/bash

ACCELERATE_CONFIG="/path/to/train_scripts/configs/muti-gpu.yaml"
TRAIN_SCRIPT="train_flux_lora_ui_with_mask_object_removal_plus.py"

PARAMS=(
    --seed=42
    --mixed_precision="bf16"
    --report_to="wandb"
    --lr_warmup_steps=0
    --output_dir="/mnt/data/zuihua_data/code/train_scripts/ouput/flux-fill-lora"
    --save_name="1000_lr_1e-3_r32_adamw_bz16_n4"
    --project_name="Object_Removal_new"
    --train_data_dir="/path/to/dataset"
    --optimizer="AdamW"
    --lr_scheduler="constant"
    --learning_rate=0.0008
    --train_batch_size=16
    --repeats=5
    --gradient_accumulation_steps=1
    --num_train_epochs=20
    --save_model_epochs=5
    --validation_epochs=1
    --rank=32
    --gradient_checkpointing
    --validation_ratio=0.1
    --pretrained_model_name_or_path="/path/to/models/FLUX.1-Fill-dev"
    --resolution=1024
    --caption_dropout=0
    --cosine_restarts=1
    --max_time_steps=0
    --blocks_to_swap=5
    --mask_dropout=0
    --reg_ratio=0.7
)

[ -n "$MODEL_PATH" ] && PARAMS+=(--model_path="$MODEL_PATH")
[ -n "$RESUME_FROM_CHECKPOINT" ] && PARAMS+=(--resume_from_checkpoint="$RESUME_FROM_CHECKPOINT")

accelerate launch \
    --config_file $ACCELERATE_CONFIG \
    $TRAIN_SCRIPT \
    "${PARAMS[@]}"
