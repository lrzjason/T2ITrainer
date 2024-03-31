#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# this is a practice codebase, mainly inspired from diffusers train_text_to_image_sdxl.py
# this codebase mainly to get the training working rather than many option to set
# therefore, it would assume something like fp16 vae fixed and baked in model, etc
# some option, me doesn't used in training wouldn't implemented like ema, etc

import argparse
import functools
import gc
import logging
import math
import os
import random
import shutil
from pathlib import Path

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import diffusers


from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from packaging import version
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
)
from pathlib import Path
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from diffusers import StableDiffusionXLPipeline
from tqdm import tqdm 
from PIL import Image 
from compel import Compel, ReturnedEmbeddingsType

import json

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.27.0.dev0")

logger = get_logger(__name__)

DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}

def compute_embeddings(batch, vae, compel, proportion_empty_prompts, caption_column,recreate=False):
    path = batch.pop("path")[0]
    full_path = os.path.join(args.train_data_dir, f"{path}.npz")
    # check if file exists
    if os.path.exists(full_path):
        # load file via torch
        try:
            if recreate:
                # remove the cache
                os.remove(full_path)
            else:
                embedding = torch.load(full_path)
                return embedding
        except Exception as e:
            print(e)
            print(f"{full_path} is corrupted, regenerating...")
    

    images = batch.pop("pixel_values")
    pixel_values = torch.stack(list(images))
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    # print('vae.dtype',vae.dtype)
    pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)

    with torch.no_grad():
        model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = model_input * vae.config.scaling_factor

    prompt = batch[caption_column][0] if isinstance(batch[caption_column][0], str) else ""
    if random.random() < proportion_empty_prompts:
        prompt = ""
    prompt_embeds, pooled_prompt_embeds = compel(prompt)
    
    latent = {
        "model_input": model_input.cpu(),
        "prompt_embeds": prompt_embeds.cpu(), 
        "pooled_prompt_embeds": pooled_prompt_embeds.cpu()
    }
    # save latent to cache file
    torch.save(latent, full_path)
    # print(f"Saved latent to {full_path}")
    return latent


def generate_timestep_weights(args, num_timesteps):
    weights = torch.ones(num_timesteps)

    # Determine the indices to bias
    num_to_bias = int(args.timestep_bias_portion * num_timesteps)

    if args.timestep_bias_strategy == "later":
        bias_indices = slice(-num_to_bias, None)
    elif args.timestep_bias_strategy == "earlier":
        bias_indices = slice(0, num_to_bias)
    elif args.timestep_bias_strategy == "range":
        # Out of the possible 1000 timesteps, we might want to focus on eg. 200-500.
        range_begin = args.timestep_bias_begin
        range_end = args.timestep_bias_end
        if range_begin < 0:
            raise ValueError(
                "When using the range strategy for timestep bias, you must provide a beginning timestep greater or equal to zero."
            )
        if range_end > num_timesteps:
            raise ValueError(
                "When using the range strategy for timestep bias, you must provide an ending timestep smaller than the number of timesteps."
            )
        bias_indices = slice(range_begin, range_end)
    else:  # 'none' or any other string
        return weights
    if args.timestep_bias_multiplier <= 0:
        return ValueError(
            "The parameter --timestep_bias_multiplier is not intended to be used to disable the training of specific timesteps."
            " If it was intended to disable timestep bias, use `--timestep_bias_strategy none` instead."
            " A timestep bias multiplier less than or equal to 0 is not allowed."
        )

    # Apply the bias
    weights[bias_indices] *= args.timestep_bias_multiplier

    # Normalize
    weights /= weights.sum()

    return weights


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Practice of a training script.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="diffusion-dpo-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        required=True,
        help="Single file .ckpt or .safetensors model file",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )

    
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )

    
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )

    
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )

    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )

    
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )

    
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )

    parser.add_argument("--num_train_epochs", type=int, default=20)
    
    
    parser.add_argument(
        "--timestep_bias_strategy",
        type=str,
        default="none",
        choices=["earlier", "later", "range", "none"],
        help=(
            "The timestep bias strategy, which may help direct the model toward learning low or high frequency details."
            " Choices: ['earlier', 'later', 'range', 'none']."
            " The default is 'none', which means no bias is applied, and training proceeds normally."
            " The value of 'later' will increase the frequency of the model's final training timesteps."
        ),
    )
    parser.add_argument(
        "--timestep_bias_multiplier",
        type=float,
        default=1.0,
        help=(
            "The multiplier for the bias. Defaults to 1.0, which means no bias is applied."
            " A value of 2.0 will double the weight of the bias, and a value of 0.5 will halve it."
        ),
    )
    parser.add_argument(
        "--timestep_bias_begin",
        type=int,
        default=0,
        help=(
            "When using `--timestep_bias_strategy=range`, the beginning (inclusive) timestep to bias."
            " Defaults to zero, which equates to having no specific bias."
        ),
    )
    parser.add_argument(
        "--timestep_bias_end",
        type=int,
        default=1000,
        help=(
            "When using `--timestep_bias_strategy=range`, the final timestep (inclusive) to bias."
            " Defaults to 1000, which is the number of timesteps that Stable Diffusion is trained on."
        ),
    )
    parser.add_argument(
        "--timestep_bias_portion",
        type=float,
        default=0.25,
        help=(
            "The portion of timesteps to bias. Defaults to 0.25, which 25% of timesteps will be biased."
            " A value of 0.5 will bias one half of the timesteps. The value provided for `--timestep_bias_strategy` determines"
            " whether the biased portions are in the earlier or later timesteps."
        ),
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    
    parser.add_argument(
        "--save_name",
        type=str,
        default="test",
        help=(
            'Save model name start with this'
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )


    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    return args



# training main function
def main(args):
    args.seed = 1234
    args.output_dir = 'F:/learning/accelerator_sdxl_train/output'
    args.logging_dir = 'logs'
    args.model_path = 'F:/models/Stable-diffusion/sdxl/b9/openxl2_b9.safetensors'
    args.gradient_accumulation_steps = 1
    args.mixed_precision = "fp16"
    # args.report_to = "tensorboard"
    
    args.report_to = "wandb"
    args.enable_xformers_memory_efficient_attention = True
    args.gradient_checkpointing = True
    args.allow_tf32 = True
    # args.learning_rate = 1e-4
    args.learning_rate = 1e-5
    args.scale_lr = False
    args.use_8bit_adam = True
    args.adam_beta1 = 0.9
    # args.adam_beta2 = 0.999
    args.adam_beta2 = 0.99

    args.adam_weight_decay = 1e-2
    args.adam_epsilon = 1e-08
    # args.train_data_dir = 'F:/ImageSet/training_script_testset_sdxl_train_validation/train/test'
    args.train_data_dir = 'F:/ImageSet/training_script_cotton_doll/train'
    args.dataset_name = None
    args.cache_dir = None
    args.caption_column = None
    args.image_column = 'image'
    args.caption_column = 'text'
    args.resolution = 1024
    args.center_crop = False
    args.max_train_samples = None
    args.proportion_empty_prompts = 0
    args.train_batch_size = 1
    args.dataloader_num_workers = 0
    args.max_train_steps = None
    args.num_train_epochs = 5
    args.lr_scheduler = "constant"
    args.lr_warmup_steps = 50

    args.timestep_bias_portion = 0.25
    args.timestep_bias_end = 1000
    args.timestep_bias_begin = 0
    args.timestep_bias_multiplier = 1.0
    args.timestep_bias_strategy = "none"
    args.max_grad_norm = 1.0
    args.save_name = "test_run"
    args.checkpointing_steps = 100
    args.validation_prompt = "cotton doll, chef_hat, hat, :3, no_humans, solo, looking_at_viewer, blush_stickers, sitting, buttons, chef, "
    args.validation_epochs = 5
    args.num_validation_images = 1

    vae_path = "F:/models/VAE/sdxl_vae.safetensors"

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    logging_dir = Path(args.output_dir, args.logging_dir)
    # create accelerator
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    
    set_seed(args.seed)


    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    pipeline = StableDiffusionXLPipeline.from_single_file(
    args.model_path,variant=weight_dtype, use_safetensors=True, 
    torch_dtype=weight_dtype).to(accelerator.device)
    # print('pipeline:')
    # print(pipeline)

    text_encoder_one = pipeline.text_encoder
    text_encoder_two = pipeline.text_encoder_2
    vae = pipeline.vae
    
    # vae = AutoencoderKL.from_single_file(
    #     vae_path
    # )
    unet = pipeline.unet

    # Load scheduler and models
    # from kohya ss sdxl_train.py 
    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
    )
    
    # Freeze vae 
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    # Move unet and text_encoders to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    unet = accelerator.prepare(unet)
    # Assume fp16 vae got fixed and baked in model
    # exception handling would be added afterward
    # vae.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    
    # Set unet as trainable.
    unet.train()

    if accelerator.mixed_precision == "fp16":
        # from kohya_ss train_util
        org_unscale_grads = accelerator.scaler._unscale_grads_
        def _unscale_grads_replacer(optimizer, inv_scale, found_inf, allow_fp16):
            return org_unscale_grads(optimizer, inv_scale, found_inf, True)

        accelerator.scaler._unscale_grads_ = _unscale_grads_replacer


    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)
    
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use paged 8-bit Adam for more lower memory usage or to fine-tune the model
    # if args.use_8bit_adam:
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use paged 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
        # optimizer_class = bnb.optim.PagedAdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    
    # Optimizer creation
    params_to_optimize = unet.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # ==========================================================
    # Create train dataset
    # ==========================================================
    data_files = {}
    # this part need more work afterward, you need to prepare 
    # the train files and val files split first before the training
    if args.train_data_dir is not None:
        data_files["train"] = os.path.join(args.train_data_dir, "**")
        def read_caption(folder,filename):
            # assume the caption file is the same as the image file
            # and ext is .txt
            # caption_file = file.replace('.jpg', '.txt')
            prompt = open(os.path.join(folder, f'{filename}.txt'), encoding='utf-8').read()
            dirname = os.path.basename(folder)
            return f'{{"file_name": "{dirname}/{file}","text":"{prompt}","path": "{dirname}/{filename}"}}\n'

        # create metadata.jsonl if not exist
        if 'metadata.jsonl' not in os.listdir(args.train_data_dir):
            metadata_file = open(os.path.join(args.train_data_dir, 'metadata.jsonl'), 'w')
            for item in os.listdir(args.train_data_dir):
                # check item is dir or file
                item_path = os.path.join(args.train_data_dir, item)
                if os.path.isdir(item_path):
                    folder_path = item_path
                    for file in os.listdir(folder_path):
                        if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.webp'):
                            # get filename and ext from file
                            filename, _ = os.path.splitext(file)
                            metadata_file.write(read_caption(folder_path,filename))
                else:
                    if item.endswith('.jpg') or item.endswith('.png') or item.endswith('.webp'):
                        filename, _ = os.path.splitext(item)
                        metadata_file.write(read_caption(folder_path,filename))
            metadata_file.close()

    
    dataset = load_dataset(
        "imagefolder",
        data_files=data_files,
        cache_dir=args.cache_dir,
    )


    # print(dataset["train"][0])
    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names
    # print(column_names)

    image_column = column_names[0]
    caption_column = column_names[1]
    # print(column_names)

    # ================================================================
    # Need to re-write with bucketing
    # ================================================================
    # Preprocessing the datasets.
    # afterward, Using bucketing rather than cropping
    # train_resize = transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR)
    # train_crop = transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution)
    train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        # print('images:',images)
        # image aug
        original_sizes = []
        all_images = []
        crop_top_lefts = []
        for image in images:
            original_sizes.append((image.height, image.width))
            image = train_resize(image)
            # if args.random_flip and random.random() < 0.5:
            #     # flip
            #     image = train_flip(image)
            if args.center_crop:
                y1 = max(0, int(round((image.height - args.resolution) / 2.0)))
                x1 = max(0, int(round((image.width - args.resolution) / 2.0)))
                image = train_crop(image)
            else:
                y1, x1, h, w = train_crop.get_params(image, (args.resolution, args.resolution))
                image = crop(image, y1, x1, h, w)
            crop_top_left = (y1, x1)
            crop_top_lefts.append(crop_top_left)
            image = train_transforms(image)
            all_images.append(image)

        examples["original_sizes"] = original_sizes
        examples["crop_top_lefts"] = crop_top_lefts
        examples["pixel_values"] = all_images
        # print('examples:',examples)
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        
        train_dataset = dataset["train"].with_transform(preprocess_train)

    # ================================================================
    # End Bucketing rewrite
    # ================================================================
    

    # ================================================================
    # Create embedding
    # ================================================================
    # Let's first compute all the embeddings so that we can free up the text encoders
    # from memory. We will pre-compute the VAE encodings too.

    # # using compel for longer prompt embedding
    compel = Compel(tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2] , text_encoder=[text_encoder_one, text_encoder_two], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])

    compute_embeddings_fn = functools.partial(
        compute_embeddings,
        vae=vae,
        compel=compel,
        proportion_empty_prompts=args.proportion_empty_prompts,
        caption_column=args.caption_column,
        recreate=False
    )

    with accelerator.main_process_first():
        train_dataset = train_dataset.map(compute_embeddings_fn, 
                                          batched=True,
                                          batch_size=args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps)

    del text_encoder_one, text_encoder_two, pipeline.tokenizer, pipeline.tokenizer_2, vae
    gc.collect()
    torch.cuda.empty_cache()

    # ================================================================
    # End create embedding 
    # ================================================================
    
    def collate_fn(examples):
        model_input = torch.stack([torch.tensor(example["model_input"]) for example in examples])
        original_sizes = [example["original_sizes"] for example in examples]
        crop_top_lefts = [example["crop_top_lefts"] for example in examples]
        prompt_embeds = torch.stack([torch.tensor(example["prompt_embeds"]) for example in examples])
        pooled_prompt_embeds = torch.stack([torch.tensor(example["pooled_prompt_embeds"]) for example in examples])

        return {
            "model_input": model_input,
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "original_sizes": original_sizes,
            "crop_top_lefts": crop_top_lefts,
        }
    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune-sdxl", config=vars(args))

    # Function for unwraping if torch.compile() was used in accelerate.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # =================================================================
    # Skip resume from checkpoint
    # =================================================================
    # Potentially load in the weights and states from a previous save
    # if args.resume_from_checkpoint:
    #     if args.resume_from_checkpoint != "latest":
    #         path = os.path.basename(args.resume_from_checkpoint)
    #     else:
    #         # Get the most recent checkpoint
    #         dirs = os.listdir(args.output_dir)
    #         dirs = [d for d in dirs if d.startswith("checkpoint")]
    #         dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    #         path = dirs[-1] if len(dirs) > 0 else None

    #     if path is None:
    #         accelerator.print(
    #             f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
    #         )
    #         args.resume_from_checkpoint = None
    #         initial_global_step = 0
    #     else:
    #         accelerator.print(f"Resuming from checkpoint {path}")
    #         accelerator.load_state(os.path.join(args.output_dir, path))
    #         global_step = int(path.split("-")[1])

    #         initial_global_step = global_step
    #         first_epoch = global_step // num_update_steps_per_epoch

    # else:
    #     initial_global_step = 0
    initial_global_step = 0
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # print("before epoch start")
    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # print("loop dataloader")
            with accelerator.accumulate(unet):
                # Sample noise that we'll add to the latents
                # model_input is vae encoded image aka latent
                model_input = batch["model_input"].to(accelerator.device)
                # get latent like random noise
                noise = torch.randn_like(model_input)

                # skip noise_offset
                # if args.noise_offset:
                #     # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                #     noise += args.noise_offset * torch.randn(
                #         (model_input.shape[0], model_input.shape[1], 1, 1), device=model_input.device
                #     )
                
                bsz = model_input.shape[0]
                # print("model_input")
                # print(model_input)

                # =========================================================
                # This part need to re-write following with sd3 paper
                # Or modify it the strategy based on training step
                # In earier training, focus to learn earlier timestep
                # In later training, focus to learn later timestep
                # =========================================================
                if args.timestep_bias_strategy == "none":
                    # Sample a random timestep for each image without bias.
                    # get 0~1000 random timestep
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                    )
                else:
                    # Sample a random timestep for each image, potentially biased by the timestep weights.
                    # Biasing the timestep weights allows us to spend less time training irrelevant timesteps.
                    weights = generate_timestep_weights(args, noise_scheduler.config.num_train_timesteps).to(
                        model_input.device
                    )
                    timesteps = torch.multinomial(weights, bsz, replacement=True).long()
                
                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                # model input is the encoded image latent
                # here is doing the forward diffusion process to add noise to latent and achieve the 
                # specific timesteps of latent noise
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)


                # ===========================================================
                # need to re-write the target size
                # ============================================================
                # time ids
                def compute_time_ids(original_size, crops_coords_top_left):
                    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
                    target_size = (args.resolution, args.resolution)
                    add_time_ids = list(original_size + crops_coords_top_left + target_size)
                    # print("add_time_ids")
                    # print(add_time_ids)
                    add_time_ids = torch.tensor([add_time_ids])
                    add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
                    return add_time_ids

                add_time_ids = torch.cat(
                    [compute_time_ids(size, crop) for size, crop in zip(batch["original_sizes"], batch["crop_top_lefts"])]
                )

                # Predict the noise residual
                unet_added_conditions = {"time_ids": add_time_ids}
                prompt_embeds = batch["prompt_embeds"].to(accelerator.device)
                pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(accelerator.device)
                unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})
                model_pred = unet(
                    noisy_model_input,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs=unet_added_conditions,
                    return_dict=False,
                )[0]

                # # Get the target for loss depending on the prediction type
                # if args.prediction_type is not None:
                #     # set prediction_type of scheduler if defined
                #     noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                # if noise_scheduler.config.prediction_type == "epsilon":
                #     target = noise
                # elif noise_scheduler.config.prediction_type == "v_prediction":
                #     target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                # elif noise_scheduler.config.prediction_type == "sample":
                #     # We set the target to latents here, but the model_pred will return the noise sample prediction.
                #     target = model_input
                #     # We will have to subtract the noise residual from the prediction to get the target sample.
                #     model_pred = model_pred - noise
                # else:
                #     raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # For the simplicity, only use epsilon
                target = noise

                # For the simplicity, only use mse_loss.
                # other option would implemented afterward, like debias
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = unet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                
                # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        # if args.checkpoints_total_limit is not None:
                        #     checkpoints = os.listdir(args.output_dir)
                        #     checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        #     checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        #     # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        #     if len(checkpoints) >= args.checkpoints_total_limit:
                        #         num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                        #         removing_checkpoints = checkpoints[0:num_to_remove]

                        #         logger.info(
                        #             f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        #         )
                        #         logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        #         for removing_checkpoint in removing_checkpoints:
                        #             removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                        #             shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"{args.save_name}-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
            if global_step >= args.max_train_steps:
                break
            
        # ==================================================
        # validation part
        # ==================================================
        if accelerator.is_main_process:
            if epoch!=0 and args.validation_prompt is not None and epoch % args.validation_epochs == 0:
                print('epoch',epoch)
                print('args.validation_epochs',args.validation_epochs)
                print('epoch _ args.validation_epochs',epoch % args.validation_epochs)
                print('args.validation_prompt',args.validation_prompt)
                logger.info(
                    f"Running validation... /n Generating {args.num_validation_images} images with prompt:"
                    f" {args.validation_prompt}."
                )
                # create pipeline
                
                vae = AutoencoderKL.from_single_file(
                    vae_path
                )
                validation_pipeline = StableDiffusionXLPipeline.from_single_file(
                    args.model_path,variant=weight_dtype, use_safetensors=True, 
                    vae=vae,
                    unet=unet)
                # if args.prediction_type is not None:
                #     scheduler_args = {"prediction_type": args.prediction_type}
                #     pipeline.scheduler = pipeline.scheduler.from_config(pipeline.scheduler.config, **scheduler_args)

                validation_pipeline = validation_pipeline.to(accelerator.device, dtype=weight_dtype)
                validation_pipeline.set_progress_bar_config(disable=True)
                steps = 30
                
                compel = Compel(tokenizer=[validation_pipeline.tokenizer, validation_pipeline.tokenizer_2] , text_encoder=[validation_pipeline.text_encoder, validation_pipeline.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])

                conditioning, pooled = compel(args.validation_prompt)
                # run inference
                # generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
                # pipeline_args = {"prompt": args.validation_prompt}

                with torch.cuda.amp.autocast():
                    images = validation_pipeline(prompt_embeds=conditioning, 
                                    pooled_prompt_embeds=pooled, 
                                    num_inference_steps=steps,
                                    guidance_scale=7,
                                    width=args.resolution, 
                                    height=args.resolution
                                    ).images

                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        np_images = np.stack([np.asarray(img) for img in images])
                        tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                    if tracker.name == "wandb":
                        tracker.log(
                            {
                                "validation": [
                                    wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                                    for i, image in enumerate(images)
                                ]
                            }
                        )

                del validation_pipeline, vae
                gc.collect()
                torch.cuda.empty_cache()
        # ==================================================
        # end validation part
        # ==================================================
    
    accelerator.wait_for_everyone()
    # ==================================================
    # validation part after training
    # ==================================================
    if accelerator.is_main_process:
        # save model
        save_path = os.path.join(args.output_dir, f"{args.save_name}-{global_step}")
        accelerator.save_state(save_path)

        del pipeline,noise_scheduler
        gc.collect()
        torch.cuda.empty_cache()

        unet = unwrap_model(unet)

        vae = AutoencoderKL.from_single_file(
            vae_path
        )
        validation_pipeline = StableDiffusionXLPipeline.from_single_file(
            args.model_path,variant=weight_dtype, use_safetensors=True, 
            vae=vae,
            unet=unet)
        # if args.prediction_type is not None:
        #     scheduler_args = {"prediction_type": args.prediction_type}
        #     pipeline.scheduler = pipeline.scheduler.from_config(pipeline.scheduler.config, **scheduler_args)

        validation_pipeline = validation_pipeline.to(accelerator.device, dtype=weight_dtype)
        validation_pipeline.set_progress_bar_config(disable=True)
        steps = 30
        
        compel = Compel(tokenizer=[validation_pipeline.tokenizer, validation_pipeline.tokenizer_2] , text_encoder=[validation_pipeline.text_encoder, validation_pipeline.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])

        conditioning, pooled = compel(args.validation_prompt)
        # run inference
        # generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
        # pipeline_args = {"prompt": args.validation_prompt}

        with torch.cuda.amp.autocast():
            images = validation_pipeline(prompt_embeds=conditioning, 
                            pooled_prompt_embeds=pooled, 
                            num_inference_steps=steps,
                            guidance_scale=7,
                            width=args.resolution, 
                            height=args.resolution
                            ).images

        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img) for img in images])
                tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
            if tracker.name == "wandb":
                tracker.log(
                    {
                        "validation": [
                            wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                            for i, image in enumerate(images)
                        ]
                    }
                )

        del validation_pipeline
        gc.collect()
        torch.cuda.empty_cache()
    # ==================================================
    # end validation part after training
    # ==================================================

    accelerator.end_training()
        #         break
        # break


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)