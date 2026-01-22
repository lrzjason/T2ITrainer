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

from datetime import datetime
import copy
import argparse
# import functools
import gc
# import logging
import math
import os
import random
# import shutil
# from pathlib import Path

import accelerate
# import datasets
import numpy as np
import safetensors
import torch

# from diffusers.image_processor import VaeImageProcessor

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, tqdm
# from tqdm.auto import tqdm
# from transformers import AutoTokenizer, PretrainedConfig
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    # QwenImageEditPipeline
)

from flux2klein.pipeline_flux2_klein import Flux2KleinPipeline

from flux2klein.transformer_flux2 import Flux2Transformer2DModel

from flux.flux_utils import compute_loss_weighting_for_sd3, compute_density_for_timestep_sampling
# from flux.pipeline_flux_kontext import FluxKontextPipeline

from pathlib import Path
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_snr
)
from diffusers.utils import (
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.torch_utils import is_compiled_module

from tqdm import tqdm 
from PIL import Image 

from sklearn.model_selection import train_test_split

from pathlib import Path
import json

# from utils.image_utils_qwenimage import CachedJsonDataset
from utils.bucket.bucket_batch_sampler import BucketBatchSampler

# https://github.com/Lightning-AI/pytorch-lightning/blob/0d52f4577310b5a1624bed4d23d49e37fb05af9e/src/lightning_fabric/utilities/seed.py
from random import getstate as python_get_rng_state
from random import setstate as python_set_rng_state

from peft import LoraConfig, prepare_model_for_kbit_training
# from lycoris import create_lycoris, LycorisNetwork

from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

from transformers import Qwen2Tokenizer, Qwen2TokenizerFast, Qwen2VLProcessor, Qwen2_5_VLForConditionalGeneration, Qwen3ForCausalLM

if is_wandb_available():
    import wandb
    
from safetensors.torch import save_file

from utils.dist_utils import flush

import glob
import shutil
# from utils.image_utils_qwenimage import compute_text_embeddings, replace_non_utf8_characters, create_empty_embedding, crop_image,get_md5_by_path

from utils.image_utils_flux2klein import CachedJsonDataset, compute_text_embeddings, get_empty_embedding, get_latent_config, replace_non_utf8_characters, create_empty_embedding, crop_image, get_md5_by_path, create_latent_config

from torchvision import transforms

from utils.utils import find_image_files_by_regex, find_index_from_right, ToTensorUniversal, get_image_files, vae_encode_utils as vae_encode, parse_indices, print_end_signal

from utils.training_set.select_training_set import get_batch_config, get_training_set, get_dataset_batch_config
from utils.lokr_utils.adapter import get_lycoris_preset, apply_lycoris
from diffusers.models import AutoencoderKLFlux2

logger = get_logger(__name__)

def print_params_require_grad(model):
    for name, param in model.named_parameters():
        print(name,param.requires_grad)
        
def memory_stats():
    print("\nmemory_stats:\n")
    print(torch.cuda.memory_allocated()/1024**2)
    # print(torch.cuda.memory_cached()/1024**2)

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run validation every X epochs."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux-dreambooth",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    
    parser.add_argument(
        "--save_name",
        type=str,
        default="flux_",
        help=(
            "save name prefix for saving checkpoints"
        ),
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    # parser.add_argument(
    #     "--scale_lr",
    #     action="store_true",
    #     default=False,
    #     help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    # )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--cosine_restarts",
        type=int,
        default=1,
        help=(
            'for lr_scheduler cosine_with_restarts'
        ),
    )
    
    
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=50, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodidy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-02, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_d_coef",
        type=float,
        default=2,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
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
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["bf16", "fp8"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default="",
        help=(
            "train data image folder"
        ),
    )
    
    
    # parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--save_model_epochs",
        type=int,
        default=1,
        help=("Save model when x epochs"),
    )
    parser.add_argument(
        "--save_model_steps",
        type=int,
        default=-1,
        help=("Save model when x steps"),
    )
    parser.add_argument(
        "--skip_epoch",
        type=int,
        default=0,
        help=("skip val and save model before x epochs"),
    )
    parser.add_argument(
        "--skip_step",
        type=int,
        default=0,
        help=("skip val and save model before x step"),
    )
    
    # parser.add_argument(
    #     "--break_epoch",
    #     type=int,
    #     default=0,
    #     help=("break training after x epochs"),
    # )
    parser.add_argument(
        "--validation_ratio",
        type=float,
        default=0.1,
        help=("dataset split ratio for validation"),
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help=("seperate model path"),
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
        "--recreate_cache",
        action="store_true",
        help="recreate all cache",
    )
    parser.add_argument(
        "--recreate_cache_target",
        action="store_true",
        help="recreate all target cache",
    )
    parser.add_argument(
        "--recreate_cache_reference",
        action="store_true",
        help="recreate all reference cache",
    )
    parser.add_argument(
        "--recreate_cache_caption",
        action="store_true",
        help="recreate all caption cache",
    )
    parser.add_argument(
        "--caption_dropout",
        type=float,
        default=0.1,
        help=("caption_dropout ratio which drop the caption and update the unconditional space"),
    )
    parser.add_argument(
        "--mask_dropout",
        type=float,
        default=0.01,
        help=("mask_dropout ratio which replace the mask with all 0"),
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help=("seperate vae path"),
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default='512',
        help=("default: '1024', accept str: '1024', '512'"),
    )
    parser.add_argument(
        "--use_debias",
        action="store_true",
        help="Use debiased estimation loss",
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=5,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--max_time_steps",
        type=int,
        default=1000,
        help="Max time steps limitation. The training timesteps would limited as this value. 0 to max_time_steps",
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "logit_snr"],
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--freeze_transformer_layers",
        type=str,
        default='',
        help="Stop training the transformer layers included in the input using ',' to seperate layers. Example: 5,7,10,17,18,19"
    )
    parser.add_argument(
        "--freeze_single_transformer_layers",
        type=str,
        default='',
        help="Stop training the transformer layers included in the input using ',' to seperate layers. Example: 5,7,10,17,18,19"
    )
    parser.add_argument(
        "--lora_layers",
        type=str,
        default=None,
        help=(
            'The transformer modules to apply LoRA training on. Please specify the layers in a comma seperated. E.g. - "to_k,to_q,to_v,to_out.0" will result in lora training of attention layers only'
        ),
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1,
        help="the FLUX.1 dev variant is a guidance distilled model. default 1 to preserve distillation.",
    )
    # parser.add_argument(
    #     "--use_fp8",
    #     action="store_true",
    #     help="Use fp8 model",
    # )
    parser.add_argument(
        "--blocks_to_swap",
        type=int,
        default=10,
        help="Suggest to 10-20 depends on VRAM",
    )
    parser.add_argument(
        "--noise_offset",
        type=float,
        default=0.01,
        help="noise offset in initial noise",
    )
    parser.add_argument(
        "--reg_ratio",
        type=float,
        default=0.0,
        help="As regularization of objective transfer learning. Set as 1 if you aren't training different objective.",
    )
    parser.add_argument(
        "--reg_timestep",
        type=int,
        default=0,
        help="As regularization of objective transfer learning. You could try different value.",
    )
    
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.json",
        help="Path to the config file.",
    )
        # default="config_qwen_edit_pairs.json",
    parser.add_argument(
        "--use_two_captions",
        action="store_true",
        help="Use _T caption and _R caption to train each direction",
    )
    parser.add_argument(
        "--slider_positive_scale",
        type=float,
        default=1.0,
        help="Slider Training positive target scale",
    )
    parser.add_argument(
        "--slider_negative_scale",
        type=float,
        default=-1.0,
        help="Slider Training negative target scale",
    )
    
    parser.add_argument(
        "--use_lokr",
        action="store_true",
        help="Use lokr instead of lora",
    )
    
    parser.add_argument(
        "--rank_alpha",
        type=float,
        default=2.0,
        help=("The rank_alpha of the LoRA or Lokr update matrices."),
    )
    parser.add_argument(
        "--lokr_factor",
        type=float,
        default=8.0,
        help=("The lokr factor of the Lokr matrices."),
    )
    
    parser.add_argument(
        "--use_new_rope",
        action="store_true",
        help="Use new rope style for main image + multiple reference images",
    )
    
    parser.add_argument(
        "--nln_samples",
        type=int,
        default=10,
        help=("how many noise samples to use for normalized latent noise" ),
    )
    parser.add_argument(
        "--nln_scale",
        type=float,
        default=0.02,
        help=("scale factor for normalized latent noise" ),
    )
    parser.add_argument(
        "--nln_method",
        type=str,
        default="nln_directional",
        help=("method for normalized latent noise, default, nln, nln_directional" ),
    )
    # parser.add_argument(
    #     "--use_torch_compile",
    #     action="store_true",
    #     help="use torch.compile improve performance",
    # )
    
    
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    # Load config file if provided
    if args.config_path and os.path.exists(args.config_path):
        try:
            with open(args.config_path, 'r', encoding='utf-8') as f:
                config_args = json.load(f)
            # Update args with values from config file
            # Ensure that config values override command-line arguments
            # Convert config values to the correct types if necessary
            for key, value in config_args.items():
                # if hasattr(args, key):
                # Attempt to convert value to the type of the existing argument
                arg_type = type(value)
                if arg_type == bool:
                    # Handle boolean conversion carefully
                    if isinstance(value, str):
                        if value.lower() in ('true', '1', 'yes'):
                            setattr(args, key, True)
                        elif value.lower() in ('false', '0', 'no'):
                            setattr(args, key, False)
                        else:
                            print(f"Could not convert '{value}' to boolean for argument '{key}'. Keeping default.")
                    else:
                        setattr(args, key, bool(value))
                else:
                    try:
                        setattr(args, key, arg_type(value))
                    except ValueError:
                        print(f"Could not convert '{value}' to type {arg_type.__name__} for argument '{key}'. Keeping default.")
                # else:
                #     print(f"Config file contains unknown argument: '{key}'. Ignoring.")
        except Exception as e:
            print(f"Could not load config file '{args.config_path}': {e}. Using command-line arguments.")

    print(f"Using config: {args}")
    return args, config_args

def main(args, config_args):
    training_name = "flux2klein"
    
    recreate_cache = args.recreate_cache
    recreate_cache_target = args.recreate_cache_target
    recreate_cache_reference = args.recreate_cache_reference
    recreate_cache_caption = args.recreate_cache_caption
    
    # override if recreate_cache
    if recreate_cache:
        recreate_cache_target = True
        recreate_cache_reference = True
        recreate_cache_caption = True
    
    # local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # torch.cuda.set_device(local_rank)  # ← critical step

    # dist.init_process_group(backend="nccl")
    # start using reference timestep
    # enable_traj_refs = True
    # traj_drop_rate = 0.15
    # t_low = 0.2
    # t_high = 0.85
    # min_step = 0.05
    # traj_weight = 0.4
    # pior_weight = 0.3
    # enable_traj_refs = args.enable_traj_refs
    # traj_drop_rate = args.traj_drop_rate
    # t_low = args.t_low
    # t_high = args.t_high
    # min_step = args.min_step
    # traj_weight = args.traj_weight
    # max_references = 2
    # reasoning_frame = args.reasoning_frame
    # reasoning_gamma = args.reasoning_gamma
    # pior_weight = args.pior_weight
    
    # baseline
    # traj_drop_rate = 0.15
    # t_low = 0.15
    # t_high = 0.85
    # min_step = 0.05
    # traj_weight = 0.4
    
    # low traj
    # traj_drop_rate = 0.1
    # t_low = 0.2
    # t_high = 0.85
    # min_step = 0.05
    # traj_weight = 0.3
    
    d_coef = 2.0
    # args.scale_lr = False
    use_8bit_adam = True
    adam_beta1 = 0.9
    # adam_beta2 = 0.999
    adam_beta2 = 0.99

    adam_weight_decay = 1e-2
    adam_epsilon = 1e-08
    # args.proportion_empty_prompts = 0
    dataloader_num_workers = 0
    max_train_steps = None

    max_grad_norm = 1.0
    revision = None
    variant = None
    prodigy_decouple = True
    prodigy_beta3 = None
    prodigy_use_bias_correction = True
    prodigy_safeguard_warmup = True
    prodigy_d_coef = 2
    
    lr_power = 1
    
    # this is for consistence validation. all validation would use this seed to generate the same validation set
    # val_seed = random.randint(1, 100)
    val_seed = 42
    
    transformer_subfolder = "transformer"
    
    # enable_redux_training = False
    # image_1 = "train"
    # image_2 = "ref"
    
    npz_path_key = "npz_path"
    batch_targets_key = "targets"
    batch_references_key = "references"
    batch_captions_key = "captions"
    prompt_embed_key = "prompt_embed"
    prompt_embeds_mask_key = "prompt_embeds_mask"
    prompt_embed_length_key = "prompt_embed_length"
    text_id_key = "text_id"
    latent_key = "latent"
    latent_path_key = f"{latent_key}_path"
    from_latent_path_key = f'from_{latent_key}_path'
    from_latent_key = f'from_{latent_key}'
    npsuffix = "f2k"
    cache_ext= f".np{npsuffix}"
    latent_ext= f".np{npsuffix}latent"
    dataset_configs = config_args['dataset_configs']
    # vae_scale_factor = int(2 ** len(temperal_downsample))
    # latents_mean = 0.1159 
    # latents_std = 0.3611
    # vae_config_block_out_channels = [
    #     128,
    #     256,
    #     512,
    #     512
    # ]
    # vae_scale_factor = 2 ** (len(vae_config_block_out_channels) - 1)
    
    # to avoid cache mutiple times on same embedding
    # use_same_embedding = True
    
    lr_num_cycles = args.cosine_restarts
    resolution = int(args.resolution)
    
    
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    if not os.path.exists(args.logging_dir): os.makedirs(args.logging_dir)
    
    
    # create metadata.jsonl if not exist
    # metadata_suffix = "qwen"
    metadata_suffix = training_name
    metadata_path = os.path.join(args.train_data_dir, f'metadata_{metadata_suffix}.json')
    val_metadata_path =  os.path.join(args.train_data_dir, f'val_metadata_{metadata_suffix}.json')
    
    # logging_dir = "logs"
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=args.logging_dir)
    # kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    kwargs = DistributedDataParallelKwargs()
    # run name is save name + datetime
    # run_name = f"{args.save_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )
    
    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    elif accelerator.mixed_precision == "fp8":
        weight_dtype = torch.float8_e4m3fn
    
    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    
    if args.lora_layers is not None and (args.lora_layers != "" and args.lora_layers != "all"):
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        # same as flux
        target_modules = [
            "to_k", "to_q", "to_v", "to_out.0"
        ]
    
    def collate_fn(examples):
        return examples
    
    datarows = None
    metadata_path = os.path.join(args.output_dir, f'dataset_{training_name}.json')
    val_metadata_path = os.path.join(args.output_dir, f'val_dataset_{training_name}.json')
    
    if dataset_configs is not None and len(dataset_configs) > 0:
        dataset_datarows = []
        val_dataset_datarows = []
        
        tokenizer_one = None
        processor = None
        text_encoder_one = None
        vae = None
        for dataset_config in dataset_configs:
            train_data_dir = dataset_config["train_data_dir"] if "train_data_dir" in dataset_config else args.train_data_dir
            # get train_data_dir basename as dataset_name
            dataset_name =  os.path.basename(train_data_dir)
            resolution = int(dataset_config["resolution"]) if "resolution" in dataset_config else args.resolution
            recreate_cache = True if ("recreate_cache" in dataset_config and dataset_config["recreate_cache"]) or args.recreate_cache else False
            repeats = dataset_config["repeats"] if "repeats" in dataset_config else args.repeats
            dataset_basename = os.path.basename(train_data_dir)
            
            image_configs = dataset_config['image_configs']
            image_configs_keys = list(image_configs.keys())
            if len(image_configs_keys) > 0:
                dataset_based_image = image_configs_keys[0]
                
            merge_configs = {**image_configs}
            exclude_base_image_keys = [key for key in merge_configs.keys() if key != dataset_based_image]
            
            caption_configs = dataset_config['caption_configs']
            target_configs = dataset_config['target_configs']
            reference_configs = dataset_config['reference_configs'] if "reference_configs" in dataset_config else None
            cache_image_ext = ".webp"
            batch_configs = dataset_config['batch_configs']
        
            # input_dir = train_data_dir
            # recreate_cache = args.recreate_cache
            
            root_dir = os.path.dirname(train_data_dir)
            cache_dir = os.path.join(root_dir, f"t2itrainer_cache_{training_name}")
            os.makedirs(cache_dir, exist_ok=True)
            
            subset_metadata_path = os.path.join(cache_dir, f'metadata_{dataset_basename}.json')
            val_subset_metadata_path = os.path.join(cache_dir, f'val_metadata_{dataset_basename}.json')
            
            # recreate_cache = dataset_config["recreate_cache"]
            recreate_cache_target = False if "recreate_cache_target" not in dataset_config else dataset_config["recreate_cache_target"]
            recreate_cache_reference = False if "recreate_cache_reference" not in dataset_config else dataset_config["recreate_cache_reference"]
            recreate_cache_caption = False if "recreate_cache_reference" not in dataset_config else dataset_config["recreate_cache_caption"]
            
            # override if recreate_cache
            if recreate_cache:
                recreate_cache_target = True
                recreate_cache_reference = True
                recreate_cache_caption = True
            
            if recreate_cache or recreate_cache_caption or recreate_cache_target or recreate_cache_reference:
                if accelerator.is_main_process:
                    # remove metadata and val_metadata_path
                    if os.path.exists(subset_metadata_path): os.remove(subset_metadata_path)
                    if os.path.exists(val_subset_metadata_path): os.remove(val_subset_metadata_path)
            
            SUPPORTED_IMAGE_TYPES = ['.jpg','.jpeg','.png','.webp']
            # files = glob.glob(f"{input_dir}/**", recursive=True)
            # image_files = [f for f in files if os.path.splitext(f)[-1].lower() in supported_image_types]
            image_files = [
                f for f in glob.iglob(os.path.join(train_data_dir, "**"), recursive=True)
                if os.path.isfile(f)
                and os.path.splitext(f)[1].lower() in SUPPORTED_IMAGE_TYPES
            ]

            # collect all image files and construct different lists
            image_pool = {}
            mapping = {}
            image_pairs = []
            for key in merge_configs.keys():
                image_pool[key] = []
                mapping[key] = {}
                
            # split files into different image pools
            for f in image_files:
                base_name = os.path.basename(f)
                filename, _ = os.path.splitext(base_name)
                
                for key in image_configs.keys():
                    config = image_configs[key]
                    # no suffix means single image training
                    if "suffix" in config and len(config["suffix"]) > 0:
                        suffix = config["suffix"]
                        suffix_index = find_index_from_right(filename,suffix)
                        if suffix_index > 0:
                            image_pool[key].append(f)
                    else:
                        image_pool[key].append(f)
                        
            # construct image pairs
            for key in merge_configs.keys():
                config = merge_configs[key]
                imageset = image_pool[key]
                suffix = config.get("suffix", "")
                prefix = ""
                if "prefix" in config:
                    prefix = config["prefix"]
                    
                for file in imageset:
                    base_name = os.path.basename(file)
                    filename, _ = os.path.splitext(base_name)
                    
                    if len(suffix) > 0:
                        suffix_index = find_index_from_right(filename,suffix)
                        filename_without_suffix = filename[:suffix_index]
                        
                        # handle prefix setting like mask which related to image
                        if "prefix" in config:
                            prefix = config["prefix"]
                            prefix_index = find_index_from_right(filename_without_suffix,prefix)
                            if prefix_index > 0:
                                filename_without_suffix = filename[:prefix_index]
                    else:
                        filename_without_suffix = filename
                        
                    subdir = os.path.dirname(file)
                    mapping_key = f"{subdir}/{filename_without_suffix}"
                    if not mapping_key in mapping[key]:
                        mapping[key][mapping_key] = []
                    mapping[key][mapping_key].append(file)
            
            for mapping_key in mapping[dataset_based_image]:
                for based_image in mapping[dataset_based_image][mapping_key]:
                    base_name = os.path.basename(based_image)
                    filename, _ = os.path.splitext(base_name)
                    pair = {
                        "mapping_key":mapping_key,
                        dataset_based_image: based_image,
                    }
                    # have_pair = True
                    for image_group_key in exclude_base_image_keys:
                        if mapping_key in mapping[image_group_key]:
                            if len(mapping[image_group_key][mapping_key]) > 1:
                                for pair_image in mapping[image_group_key][mapping_key]:
                                    if filename in pair_image:
                                        pair[image_group_key] = pair_image
                            else:
                                pair[image_group_key] = mapping[image_group_key][mapping_key][0]
                        # if any image doesn't have fullset pair, it should be skipped
                    #     else:
                    #         have_pair = False
                    # if have_pair:
                    #     image_pairs.append(pair)
                    image_pairs.append(pair)
            
            train_transforms = transforms.Compose([ToTensorUniversal(), transforms.Normalize([0.5], [0.5])])
                            
            def get_cache_dir(image_path):
                dir_name = os.path.dirname(image_path)
                # create subdir
                if not Path(dir_name).resolve() == Path(args.train_data_dir).resolve():
                    # get subdir name
                    subdir_name = os.path.basename(dir_name)
                    # create subdir in temp dir
                    temp_subdir_name = os.path.join(cache_dir, subdir_name)
                    os.makedirs(temp_subdir_name, exist_ok=True)
                    dir_name = temp_subdir_name
                return dir_name
        
            def cache_image(vae, train_data_dir, image_path, config, recreate_cache=False):
                # get dir name
                working_dir = get_cache_dir(image_path)
                # convert to int
                resize = int(config["resize"]) if "resize" in config else int(resolution)
                filename, image_ext = os.path.splitext(image_path)
                basename = os.path.basename(filename)
                
                resized_image_path = os.path.join(working_dir, f"{basename}_{resize}{cache_image_ext}")
                resized_latent_path = os.path.join(working_dir, f"{basename}_{resize}{latent_ext}")
                if not os.path.exists(resized_image_path) or not os.path.exists(resized_latent_path) or recreate_cache:
                    image = crop_image(image_path,resolution=resize)
                    image_height, image_width, _ = image.shape
                    
                    # convert nparray to pil image and save to subdir
                    pil_image = Image.fromarray((image).astype('uint8'))
                    pil_image.save(resized_image_path, cache_image_ext.replace('.',''))
                    
                    image = train_transforms(image)
                    latent_dict = vae_encode(vae, image, vae_type="flux")
                    torch.save(latent_dict, resized_latent_path)
                else:
                    try:
                        image = np.array(Image.open(resized_image_path).convert("RGB")).astype(np.uint8)
                    except Exception as e:
                        print(f"Error loading image {resized_image_path}: {e}")
                        image = crop_image(image_path,resolution=resize)
                        # convert nparray to pil image and save to subdir
                        pil_image = Image.fromarray((image).astype('uint8'))
                        pil_image.save(resized_image_path, cache_image_ext.replace('.',''))
                        image = train_transforms(image)
                        latent_dict = vae_encode(vae, image, vae_type="flux")
                        torch.save(latent_dict, resized_latent_path)
                    image_height, image_width, _ = image.shape
                
                bucket = f"{image_width}x{image_height}"
                result = { 
                    "bucket": bucket, 
                    "image_path": resized_image_path, 
                    "original_image_path": image_path,
                    "latent_path": resized_latent_path
                }
                return result
                
            # construct targets/references list
            def construct_image_list(vae, train_data_dir, image_pair, subdir_caches, configs, recreate_cache):
                bucket = None
                image_set = {}
                for config_key in configs.keys():
                    image_configs = configs[config_key]
                    image_list = []
                    for config in image_configs:
                        # sample reference list
                        sample_type = "from_same_name" if "sample_type" not in config else config["sample_type"]
                        assert "image" in config, "image key not in reference config"
                        if sample_type == "from_same_name":
                            image_key = config["image"]
                            image_path = image_pair[image_key]
                            image_obj = cache_image(vae, train_data_dir, image_path, config, recreate_cache)
                            if bucket is None:
                                bucket = image_obj["bucket"]
                            
                            if "from_image" in config:
                                from_image_key = config["from_image"]
                                from_image_path = image_pair[from_image_key]
                                from_image_obj = cache_image(vae, train_data_dir, from_image_path, config, recreate_cache)
                                # merge two dict
                                image_obj["from_image_path"] = from_image_obj["image_path"]
                                image_obj["from_latent_path"] = from_image_obj["latent_path"]
                            
                            image_list.append(image_obj)
                            # image_list.append(image_path)
                        elif sample_type == "from_subdir":
                            image_key = config["image"]
                            assert "suffix" in config, "suffix key not in reference config"
                            suffix = config["suffix"]
                            count = config["count"] if "count" in config else 1
                            # get all images in the same subdir with same suffix
                            base_image_path = image_pair[image_key]
                            base_image_dir = os.path.dirname(base_image_path)
                            # reuse cached results
                            if base_image_dir in subdir_caches:
                                reference_files = subdir_caches[base_image_dir]
                            else:
                                # use glob to find all images with same suffix in the subdir
                                reference_files = get_image_files(base_image_dir)
                                subdir_caches[base_image_dir] = reference_files
                            excluded_reference_files = find_image_files_by_regex(reference_files, f".*{suffix}.*")
                            # remove base_image_path from reference_files
                            excluded_reference_files = [f for f in excluded_reference_files if Path(f).resolve() != Path(base_image_path).resolve()]
                            # shuffle reference files
                            random.shuffle(excluded_reference_files)
                            ref_list = []
                            # use count, dropout, min_sample to randomly select reference images
                            for ref_file in excluded_reference_files:
                                if len(ref_list) >= count:
                                    break
                                image_obj = cache_image(vae, train_data_dir, ref_file, config, recreate_cache)
                                ref_list.append(image_obj)
                            image_list += ref_list
                    image_set[config_key] = image_list
                return image_set, bucket
        
            @torch.no_grad()
            def cache_process(dataset_name, cache_datarows, validation_datarows, 
                              image_pairs, train_data_dir, recreate_cache_target, 
                              recreate_cache_reference, recreate_cache_caption,
                              vae, tokenizer_one, text_encoder_one, processor
                              ):
                # datarows = []
                if len(image_pairs) > 0:
                    # full_datarows = []
                    recache = recreate_cache
                    if os.path.exists(subset_metadata_path) and (not recreate_cache and not recreate_cache_target and not recreate_cache_reference and not recreate_cache_caption):
                        try:
                            with open(subset_metadata_path, "r", encoding='utf-8') as readfile:
                                cache_datarows += json.loads(readfile.read(), strict=False)
                                
                            if os.path.exists(val_subset_metadata_path):
                                with open(val_subset_metadata_path, "r", encoding='utf-8') as readfile:
                                    validation_datarows += json.loads(readfile.read(), strict=False)
                            
                        except Exception as e:
                            # remove metadata file
                            os.remove(subset_metadata_path)
                            recache = True
                            print(f"Error loading metadata {subset_metadata_path}: {e}")
                    else:
                        recache = True
                    if recache:
                        
                        if vae is None:
                            vae = AutoencoderKLFlux2.from_pretrained(
                                args.pretrained_model_name_or_path,
                                subfolder="vae",
                            )
                            vae.requires_grad_(False)
                            vae.to(accelerator.device, dtype=torch.float32)
                        
                        if tokenizer_one is None:
                            # Offload models to CPU and load necessary components
                            tokenizer_one = Qwen2TokenizerFast.from_pretrained(
                                args.pretrained_model_name_or_path,
                                subfolder="tokenizer",
                            )

                        # if processor is None:
                        #     processor = Qwen2VLProcessor.from_pretrained(
                        #         args.pretrained_model_name_or_path,
                        #         subfolder = 'tokenizer',
                        #     )
                        if text_encoder_one is None:
                            text_encoder_one = Qwen3ForCausalLM.from_pretrained(
                                args.pretrained_model_name_or_path, subfolder="text_encoder"
                            )
                            text_encoder_one.requires_grad_(False)
                            text_encoder_one.to(accelerator.device, dtype=weight_dtype)
                        
                        tokenizers = [tokenizer_one]
                        text_encoders = [text_encoder_one]
                        if accelerator.is_main_process:
                            create_empty_embedding(tokenizers,text_encoders)
                            create_latent_config(vae)
                            
                        embedding_objects = {
                            "dataset": dataset_name
                        }
                        subdir_caches = {}
                        
                        vae.to(accelerator.device, dtype=torch.float32)
                        print(f"Start caching {len(image_pairs)} latents...")
                        # first loop to cache all images and latents
                        for image_pair in tqdm(image_pairs):
                            embedding_object = {}
                            embedding_object[batch_targets_key] = {}
                            embedding_object[batch_references_key] = {}
                            
                            target_set, target_bucket = construct_image_list(vae, train_data_dir, image_pair, subdir_caches, target_configs, recreate_cache_target)
                            embedding_object[batch_targets_key] = target_set
                            embedding_object["bucket"] = target_bucket
                            
                            if reference_configs is not None:
                                reference_set, _ = construct_image_list(vae, train_data_dir, image_pair, subdir_caches, reference_configs, recreate_cache_reference)
                                embedding_object[batch_references_key] = reference_set
                                
                            mapping_key = image_pair["mapping_key"]
                            embedding_objects[mapping_key] = embedding_object
                            
                            # clear memory
                            flush()
                        vae.to("cpu")
                        flush()
                        accelerator.wait_for_everyone()
                        
                        text_encoder_one.to(accelerator.device, dtype=weight_dtype)
                        print(f"Start caching {len(image_pairs)} embeddings...")
                        for image_pair in tqdm(image_pairs):
                            mapping_key = image_pair["mapping_key"]
                            embedding_object = embedding_objects[mapping_key]
                        # second loop to cache all text embeddings
                        # for mapping_key, embedding_object in tqdm(embedding_objects.items()):
                            target_cache_dir = get_cache_dir(mapping_key)
                            basename = os.path.basename(mapping_key)
                            json_file = os.path.join(target_cache_dir, f"{basename}.json")
                            
                            if os.path.exists(json_file) and not recreate_cache:
                                cache_datarows.append({
                                    "json_path": json_file,
                                    "bucket": f"{embedding_objects['dataset']}_{embedding_object['bucket']}",
                                    "dataset": embedding_objects["dataset"]
                                })
                                continue
                            
                            embedding_object[batch_captions_key] = {}
                            for caption_config_key in caption_configs.keys():
                                outter_drop_index = None
                                drop_index = None
                                
                                caption_config = caption_configs[caption_config_key]
                                # image_target = caption_config_key
                                image_target = caption_config["image"]
                                
                                # default dropout is 1 to not use image ref for training
                                # ref_image_dropout = 0
                                    
                                image_path = image_pair[image_target]
                                working_dir = get_cache_dir(image_path)
                                filename = os.path.basename(image_path)
                                folder_path = os.path.dirname(image_path)
                                # get filename and ext from file
                                filename, _ = os.path.splitext(filename)
                                json_obj = { }
                                # read caption
                                # caption_ext = '.txt'
                                caption_ext = caption_config['ext']
                                text_path = os.path.join(folder_path, f'{filename}{caption_ext}')
                                content = ''

                                file_path = os.path.join(working_dir, filename)
                                npz_path = f'{file_path}_{caption_config_key}{cache_ext}'
                                json_obj[npz_path_key] = npz_path
                                
                                if os.path.exists(text_path):
                                    json_obj["text_path"] = text_path
                                    try:
                                        content = open(text_path, encoding='utf-8').read()
                                    except:
                                        content = open(text_path, encoding='utf-8').read()
                                        # try to remove non utf8 character
                                        content = replace_non_utf8_characters(content)
                                
                                if recreate_cache_caption or not os.path.exists(npz_path):
                                    # image_list = []
                                    if "reference_list" in caption_config:
                                        reference_list_config = caption_config["reference_list"]
                                        reference_key = reference_list_config["reference_config"]
                                        # default resize is 384
                                        # resize = int(reference_list_config["resize"]) if "resize" in reference_list_config else int(resolution)
                                        # dropout = reference_list_config["dropout"] if "dropout" in reference_list_config else 0.0
                                        # min_length = reference_list_config["min_length"] if "min_length" in reference_list_config else 0
                                        references = embedding_object["references"]
                                        references_list = references[reference_key] if reference_key in references else []
                                        if len(references_list) == 0:
                                            print(f"Warning: no references found for reference key {reference_key} in image pair {mapping_key}")
                                        # if dropout is 0.1, random is 0.2
                                        # it means use image as reference
                                        # if dropout is 0.2, random is 0.1
                                        # it means use only text as reference
                                        # because dropout is cache, each recreate cache will have different reference
                                        
                                        # for ref_image_obj in references_list:
                                        #     # use original image as reference path avoid double compression
                                        #     original_image_path = ref_image_obj["original_image_path"]
                                        #     if dropout < random.random() or len(image_list) <= min_length:
                                        #         image = crop_image(original_image_path,resolution=resize)
                                        #         image_list.append(image)
                                        
                                    prompt_embeds, prompt_embeds_mask, prompt_embed_length, _ = compute_text_embeddings(
                                        text_encoders,
                                        tokenizers,
                                        content,
                                        device=text_encoders[0].device,
                                        # image=image_list,
                                        image=None,
                                        processor=None,
                                        instruction=caption_config["instruction"] if "instruction" in caption_config else None,
                                        drop_index=outter_drop_index
                                    )
                                    # use same drop index when same caption config
                                    if outter_drop_index is None and drop_index is not None:
                                        outter_drop_index = drop_index
                                        
                                    prompt_embed = prompt_embeds.squeeze(0)
                                    # text_id = text_ids.squeeze(0)
                                    npz_dict = {
                                        prompt_embed_key: prompt_embed.cpu(), 
                                        prompt_embeds_mask_key: prompt_embeds_mask.cpu(),
                                        prompt_embed_length_key: prompt_embed_length.cpu(),
                                    }
                                    
                                    torch.save(npz_dict, npz_path)
                                    
                                    del npz_dict, prompt_embeds, prompt_embeds_mask, prompt_embed_length
                                embedding_object[batch_captions_key][caption_config_key] = json_obj
                                
                            # save embedding_object as a json file
                            with open(json_file, "w") as f:
                                json.dump(embedding_object, f, indent=4)
                            
                            cache_datarows.append({
                                "json_path": json_file,
                                "bucket": f"{embedding_objects['dataset']}_{embedding_object['bucket']}",
                                "dataset": embedding_objects["dataset"]
                            })
                        
                        text_encoder_one.to("cpu")
                        # , processor, prompt_embed
                        
                        # del image_pairs_dataset, image_pairs_sampler, image_pairs_dataloader
                        flush()
                
                # result = datarows.copy()
                # del datarows, image_pairs, embedding_objects
                accelerator.free_memory()  # Critical for HF components
                torch.cuda.synchronize()  # Ensure all GPU ops complete
                flush()
                # return result

            cache_datarows = []
            validation_datarows = []
            cache_process(dataset_name, cache_datarows, validation_datarows, 
                          image_pairs, train_data_dir, recreate_cache_target,
                          recreate_cache_reference, recreate_cache_caption,
                          vae, tokenizer_one, text_encoder_one, processor)
        
            # Handle validation split
            if args.validation_ratio > 0 and not os.path.exists(val_subset_metadata_path):
                train_ratio = 1 - args.validation_ratio
                validation_ratio = args.validation_ratio
                if len(cache_datarows) == 1:
                    cache_datarows = cache_datarows + cache_datarows.copy()
                    validation_ratio = 0.5
                    train_ratio = 0.5
                training_datarows, validation_datarows = train_test_split(
                    cache_datarows,
                    train_size=train_ratio,
                    test_size=validation_ratio
                )
                cache_datarows = training_datarows.copy()
            
            
            if not os.path.exists(subset_metadata_path) or recreate_cache:
                if len(cache_datarows) > 0:
                    with open(subset_metadata_path, "w", encoding='utf-8') as outfile:
                        outfile.write(json.dumps(cache_datarows))
            
            # Save validation metadata
            if len(validation_datarows) > 0 or recreate_cache:
                with open(val_subset_metadata_path, "w", encoding='utf-8') as outfile:
                    outfile.write(json.dumps(validation_datarows))
                # Clear memory
                # del training_datarows, full_datarows

            # multiply dataset datarows by dataset repeats
            dataset_datarows += cache_datarows * repeats
            val_dataset_datarows += validation_datarows
        
        
        # Save updated metadata
        with open(metadata_path, "w", encoding='utf-8') as outfile:
            outfile.write(json.dumps(dataset_datarows))
        # Save updated metadata
        with open(val_metadata_path, "w", encoding='utf-8') as outfile:
            outfile.write(json.dumps(val_dataset_datarows))

        # clear models
        del vae, tokenizer_one, processor, text_encoder_one
        
    # if accelerator.is_main_process:
    # load datarows from metadata_path
    if datarows is None:
        with open(metadata_path, "r", encoding='utf-8') as readfile:
            datarows = json.loads(readfile.read(), strict=False)
        
    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    elif accelerator.mixed_precision == "fp8":
        weight_dtype = torch.float8_e4m3fn
    
    offload_device = accelerator.device
        
    if not (args.model_path is None or args.model_path == ""):
        # config = f"{args.pretrained_model_name_or_path}/transformer/config.json"
        transformer = Flux2Transformer2DModel.from_single_file(args.model_path, 
                            # config=config,  
                            torch_dtype=weight_dtype
                        ).to(offload_device)
    else:
        transformer_folder = os.path.join(args.pretrained_model_name_or_path, transformer_subfolder)
        # weight_file = "diffusion_pytorch_model"
        variant = None
        transformer = Flux2Transformer2DModel.from_pretrained(
                    transformer_folder, variant=variant,  
                    torch_dtype=weight_dtype
                ).to(offload_device)
    
    # set rope style
    # transformer.select_rope(args.use_new_rope)
    flush()

    if "quantization_config" in transformer.config:
        transformer = prepare_model_for_kbit_training(transformer, use_gradient_checkpointing=False)
    else:
        transformer = transformer.to(offload_device, dtype=weight_dtype)
        transformer.requires_grad_(False)
    
    # if args.use_torch_compile:
    #     print("\nCompiling the model...")
    #     # Compile the model
    #     transformer = torch.compile(transformer, mode="max-autotune")
     
    # print("\nCompile completed.")
    # is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0
    # if is_swapping_blocks:
    #     # Swap blocks between CPU and GPU to reduce memory usage, in forward and backward passes.
    #     logger.info(f"enable block swap: blocks_to_swap={args.blocks_to_swap}")
    #     transformer.enable_block_swap(args.blocks_to_swap, accelerator.device)
    
    if args.use_lokr:
        preset = get_lycoris_preset()
        print("\nDefined LyCORIS preset:")
        print(preset)

        # --- 3. Apply LyCORIS to the model ---
        lycoris_net = apply_lycoris(
            transformer,
            multiplier=1.0,
            preset=preset,
            rank=args.rank,      # Specify the rank (dim) for LoKr
            alpha=args.rank_alpha,     # Specify the alpha for LoKr
            factor=args.lokr_factor,   # Not used when rank is specified
            algo="lokr",  # Use LoKr for all targeted layers
        )
        lycoris_net.to(accelerator.device)
    else:
        print("\nAdding LoRA adapters to the model...")
        # print gpu rank
        print(f"GPU rank: {accelerator.process_index}")
        # now we will add new LoRA weights to the attention layers
        transformer_lora_config = LoraConfig(
            # use_dora=args.use_dora,
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            # target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            target_modules=target_modules,
        )
        transformer.add_adapter(transformer_lora_config)
        
        
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # default to skip 59 layer, 59 layer mainly control texture and it is very easy to destroy while training.
    # freezed_layers = [59]
    freezed_layers = []
    if args.freeze_transformer_layers is not None and args.freeze_transformer_layers != '':
        freezed_layers = parse_indices(args.freeze_transformer_layers)
    
    freezed_single_layers = []
    if args.freeze_single_transformer_layers is not None and args.freeze_single_transformer_layers != '':
        freezed_single_layers = parse_indices(args.freeze_single_transformer_layers)
    
    # if args.use_lokr:
    #     # exclude last layer for lokr training to avoid horizontal lines
    #     freezed_layers.append(59)
    print("freezed_layers: ", freezed_layers)
    print("freezed_single_layers: ", freezed_single_layers)
    # Freeze the layers
    for name, param in transformer.named_parameters():
        if "single_transformer_" in name:
            if 'model.' in name:
                name = name.replace('model.', '')
            if '_orig_mod.' in name:
                name = name.replace('_orig_mod.', '')
            name_split = name.split(".")
            layer_order = name_split[1]
            if int(layer_order) in freezed_single_layers:
                param.requires_grad = False
        elif "transformer" in name:
            if 'model.' in name:
                name = name.replace('model.', '')
            if '_orig_mod.' in name:
                name = name.replace('_orig_mod.', '')
            name_split = name.split(".")
            layer_order = name_split[1]
            if int(layer_order) in freezed_layers:
                param.requires_grad = False
                
    if args.use_lokr:
        for name, param in lycoris_net.named_parameters():
            if "transformer" in name:
                name_split = name.split("blocks_")
                suffix = name_split[1]
                suffix_split = suffix.split("_")
                layer_order = suffix_split[0]
                if int(layer_order) in freezed_layers:
                    param.requires_grad = False
                # print(name,"param.requires_grad",param.requires_grad)
            
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_model_lokr_hook(models, weights, output_dir):
        print("save process...")
        for model in models:
            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()
            
        if accelerator.is_main_process:
            last_part = os.path.basename(os.path.normpath(output_dir))
            file_path = f"{output_dir}/{last_part}.safetensors"
            lycoris_net.save_weights(file_path, torch.bfloat16, metadata={"source": "T2ITrainer"})
                
            # save config to output dir using shutil
            if args.config_path:
                # save config to output dir for reproduce
                shutil.copy(args.config_path, output_dir)
            
            # save training_layout_configs, captions_selection and dataset_configs to output dir using json
            # create one json to save all configs
            # configs = {
            #     "training_set": training_set,
            #     "dataset_configs": dataset_configs,
            # }
            # configs_file = os.path.join(output_dir, "config_details.json")
            # with open(configs_file, "w", encoding='utf-8') as outfile:
            #     outfile.write(json.dumps(configs, indent=4, ensure_ascii=False))
                  
    def load_model_lokr_hook(input_dir):
        # transformer_ = None
        # while len(models) > 0:
        #     model = models.pop()
        #     if isinstance(model, type(unwrap_model(transformer))):
        #         transformer_ = model
        #     else:
        #         raise ValueError(f"unexpected save model: {model.__class__}")

        # Load the LoKr weights
        last_part = os.path.basename(os.path.normpath(input_dir))
        file_path = f"{input_dir}/{last_part}.safetensors"
        
        
        qwen_preset = get_lycoris_preset()
        lycoris_net = apply_lycoris(
            transformer,
            multiplier=1.0,
            preset=qwen_preset, # Crucial: Use the exact same preset
            rank=args.rank,      # Specify the rank (dim) for LoKr
            alpha=args.rank_alpha,     # Specify the alpha for LoKr
            factor=args.lokr_factor,   # Not used when rank is specified
            algo="lokr",  # Use LoKr for all targeted layers
        )
        load_state = lycoris_net.load_weights(file_path)
        
        return lycoris_net
    
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            
            # check models length
            # print(f"Saving models, length: {len(models)}")
            
            
            # print models keys for debug
            # for model in models:
            #     print(f"Model keys: {list(model.state_dict().keys())}")
            
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder atten layers
            transformer_lora_layers_to_save = None
            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    peft_model_state_dict = get_peft_model_state_dict(model)
                    # print(f"loaded peft model state dict: {peft_model_state_dict.keys()}")
                    transformer_lora_layers_to_save = convert_state_dict_to_diffusers(peft_model_state_dict)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            # check transformer_lora_layers_to_save is None
            if transformer_lora_layers_to_save is None:
                raise ValueError("transformer_lora_layers_to_save is None")
            
            # save all
            Flux2KleinPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save
            )
            
            last_part = os.path.basename(os.path.normpath(output_dir))
            file_path = f"{output_dir}/{last_part}.safetensors"
            ori_file = f"{output_dir}/pytorch_lora_weights.safetensors"
            if os.path.exists(ori_file): 
                # copy ori to new name
                shutil.copy(ori_file, file_path)
                
            # save config to output dir using shutil
            if args.config_path:
                # save config to output dir for reproduce
                shutil.copy(args.config_path, output_dir)
            
            # save training_layout_configs, captions_selection and dataset_configs to output dir using json
            # create one json to save all configs
            # configs = {
            #     "training_set": training_set,
            #     "dataset_configs": dataset_configs,
            # }
            # configs_file = os.path.join(output_dir, "config_details.json")
            # with open(configs_file, "w", encoding='utf-8') as outfile:
            #     outfile.write(json.dumps(configs, indent=4, ensure_ascii=False))
                
    def load_model_hook(models, input_dir):
        print("load_model_hook input_dir", input_dir)
        transformer_ = None
        while len(models) > 0:
            model = models.pop()
            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        model_file = os.path.join(input_dir, "pytorch_lora_weights.safetensors")
        lora_state_dict = safetensors.torch.load_file(model_file, device="cpu")
        # metadata = _load_sft_state_dict_metadata(model_file)
        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )
        
        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [transformer_]
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    if args.use_lokr:
       transformer_lora_parameters = list(filter(lambda p: p.requires_grad, lycoris_net.parameters()))
    else:
       transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    # Optimization parameters
    transformer_lora_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}
    params_to_optimize = [transformer_lora_parameters_with_lr]
    
    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(adam_beta1, adam_beta2),
            weight_decay=adam_weight_decay,
            eps=adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally bettere to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            d_coef=d_coef,
            betas=(adam_beta1, adam_beta2),
            beta3=prodigy_beta3,
            weight_decay=adam_weight_decay,
            eps=adam_epsilon,
            decouple=prodigy_decouple,
            use_bias_correction=prodigy_use_bias_correction,
            safeguard_warmup=prodigy_safeguard_warmup,
        )
    
    # ================================================================
    # End create embedding 
    # ================================================================
    
    # if accelerator.is_main_process:
    # create dataset based on input_dir
    # (latent - latents_mean) * latents_std
    latent_config = get_latent_config()
    latents_bn_mean = latent_config["latents_bn_mean"].to(accelerator.device)
    latents_bn_std = latent_config["latents_bn_std"].to(accelerator.device)
    train_dataset = CachedJsonDataset(datarows, 
                 latents_bn_mean, latents_bn_std, accelerator.device, weight_dtype,
                 latent_path_key, latent_key, npz_path_key, prompt_embed_key, prompt_embeds_mask_key, prompt_embed_length_key,
                 from_latent_path_key, from_latent_key)

    # referenced from everyDream discord minienglish1 shared script
    #create bucket batch sampler
    distributed_bucket_batch_sampler = BucketBatchSampler(train_dataset, batch_size=args.train_batch_size)
    #initialize the DataLoader with the bucket batch sampler
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=distributed_bucket_batch_sampler,
        collate_fn=collate_fn,
        num_workers=dataloader_num_workers,
    )
    
    # train_dataloader = accelerator.prepare(
    #     train_dataloader
    # )
    
    accelerator.wait_for_everyone()

    # ✅ Robust global step calculation — works for any num_gpus
    # total_batch_size = (
    #     args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    # )
    # num_update_steps_per_epoch = math.ceil(len(train_dataset) / total_batch_size)

    # # Only main process sets max_train_steps to avoid race (though it's read-only)
    # if max_train_steps is None:
    #     max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    #     # Broadcast to all processes (optional but safe)
    #     # max_train_steps = int(accelerator.gather(torch.tensor(max_train_steps, device=accelerator.device)).mean().item())

    # # Recompute num_train_epochs if max_train_steps was set manually
    # args.num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    train_dataloader = accelerator.prepare(train_dataloader)
    # Scheduler and math around the number of training steps.
    override_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if max_train_steps is None:
        max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        override_max_train_steps = True

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if override_max_train_steps:
        max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=max_train_steps,
        num_cycles=lr_num_cycles,
        power=lr_power,
    )

    flush()
    is_swapping_blocks = False
    if args.use_lokr:
        lycoris_net, transformer, optimizer, lr_scheduler = accelerator.prepare(lycoris_net, transformer, optimizer, lr_scheduler)
        # transformer = accelerator.prepare(transformer, device_placement=[not is_swapping_blocks])
        # transformer = accelerator.prepare(transformer)
    else:
        # transformer = accelerator.prepare(transformer, device_placement=[not is_swapping_blocks])
        transformer, optimizer, lr_scheduler = accelerator.prepare(transformer, optimizer, lr_scheduler)
    
    if accelerator.is_main_process:
        print(f"Dataset size: {len(train_dataset)}")
        print(f"Per-GPU batches per epoch: {len(train_dataloader)}")
        print(f"Global batches per epoch: {len(train_dataloader) * accelerator.num_processes}")
        print(f"Update steps per epoch: {num_update_steps_per_epoch}")
        print(f"Total training steps: {max_train_steps}")
    # print(f"[RANK {accelerator.process_index}] Prepared transformer type: {type(transformer)}")
    # print(f"[RANK {accelerator.process_index}] Unwrapped: {type(accelerator.unwrap_model(transformer))}")
    # print(f"[RANK {accelerator.process_index}] # models in accelerator: {len(accelerator._models)}")

    # unwrapped = accelerator.unwrap_model(transformer)
    # print("PEFT config keys:", list(unwrapped.peft_config.keys()) if hasattr(unwrapped, 'peft_config') else "None")  
    # torch._inductor.config.inplace_buffers = False  # 🔥 Critical fix for this KeyError
    # transformer = torch.compile(transformer)
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = f"{training_name}-lora"
        try:
            accelerator.init_trackers(tracker_name, config=vars(args))
            # wandb_tracker = accelerator.get_tracker("wandb")
            # wandb_tracker.run_name = args.save_name
        except:
            print("Trackers not initialized")
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    if accelerator.is_main_process:
        print(f"[Rank {accelerator.process_index}] "
        f"len(dataset)={len(train_dataset)}, "
        f"len(dataloader)={len(train_dataloader)}, ")
    global_step = 0
    first_epoch = 0

    if args.use_lokr:
        accelerator.register_save_state_pre_hook(save_model_lokr_hook)
        # accelerator.register_load_state_pre_hook(load_model_lokr_hook)
    else:
        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    resume_step = 0
    flush()
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint and args.resume_from_checkpoint != "":
        # if args.resume_from_checkpoint != "latest":
        #     path = os.path.basename(args.resume_from_checkpoint)
        # else:
        #     # Get the mos recent checkpoint
        #     dirs = os.listdir(args.output_dir)
        #     dirs = [d for d in dirs if d.startswith(args.save_name)]
        #     dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        #     path = dirs[-1] if len(dirs) > 0 else None

        # if path is None:
        #     accelerator.print(
        #         f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
        #     )
        #     args.resume_from_checkpoint = None
        #     initial_global_step = 0
        # else:
        path = os.path.basename(args.resume_from_checkpoint)
        accelerator.print(f"Resuming from checkpoint {path}")
        global_step = int(path.split("-")[-1])

        initial_global_step = global_step
        resume_step = global_step
        first_epoch = global_step // num_update_steps_per_epoch

        # save_dir = os.path.join(args.output_dir, path)
        save_dir = args.resume_from_checkpoint
        
        if args.use_lokr:
            lycoris_net = load_model_lokr_hook(save_dir)
            lycoris_net = accelerator.prepare(lycoris_net, device_placement=[not is_swapping_blocks])
            transformer = accelerator.prepare(transformer, device_placement=[not is_swapping_blocks])
            # transformer_lora_parameters = list(filter(lambda p: p.requires_grad, lycoris_net.parameters()))
        else:
            accelerator.load_state(save_dir)
            transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
        # Optimization parameters
        transformer_lora_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}
        params_to_optimize = [transformer_lora_parameters_with_lr]
        
        optimizer_path = f"{save_dir}/optimizer.bin"
        scheduler_path = f"{save_dir}/scheduler.bin"
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path))
        if os.path.exists(scheduler_path):
            lr_scheduler.load_state_dict(torch.load(scheduler_path))
            
    else:
        initial_global_step = 0
    
    progress_bar = tqdm(
        # range(0, max_train_steps),  # = len(train_dataloader) // grad_acc
        # only show total steps on one machine.
        range(0, max_train_steps),
        initial=initial_global_step,
        disable=not accelerator.is_local_main_process,
        desc="Steps",
    )
    
    def sample_noise(
        target_list=None,    # 目标图像 (PIL Image或tensor)
        n_samples=10,         # NLN采样数量
        scale_factor=0.02,    # 缩放因子
        method="default",
        device=accelerator.device,
        dtype=torch.float32,
    ):
        default_noise_weight = 0.3
        if method == "default" or default_noise_weight > random.random():
            return torch.randn_like(target_list).to(accelerator.device)
            # 1. 语义擦除: 生成n_candidates个随机噪声并聚合
        noises = []
        for _ in range(n_samples):
            # 生成与target_latent相同形状的随机噪声
            noise = torch.randn_like(target_list, device=device, dtype=dtype)
            noises.append(noise)
        
        # 堆叠并求和，然后归一化以保持方差
        stacked = torch.stack(noises)  # [n_candidates, B, C, T=1, H, W]
        summed = torch.sum(stacked, dim=0)  # [B, C, T=1, H, W]
        erased_noise = summed / (n_samples ** 0.5)  # 保持方差为1
        
        nln_noise_weight = 0.3
        if method == "nln" or nln_noise_weight > random.random():
            return erased_noise
        
        # 标准化目标
        target_mean = target_list.mean()
        target_std = target_list.std() + 1e-8
        target_normalized = (target_list - target_mean) / target_std
        
        # 凸组合: 向目标移动bias_strength比例
        semantic_noise = (1 - scale_factor) * erased_noise + scale_factor * target_normalized
        
        # 3. 重新标准化以保持N(0,1)分布
        noise_mean = semantic_noise.mean()
        noise_std = semantic_noise.std() + 1e-8
        semantic_noise = (semantic_noise - noise_mean) / noise_std
        
        return semantic_noise  # shape [B, C, T=1, H, W]
    
    # handle guidance
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32, mu=0.8):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        # schedule_timesteps = torch.arange(1000, 0, -1, dtype=torch.int64, device=accelerator.device)
        # schedule_timesteps is 1000 to 1
        # timesteps = timesteps.round().to(torch.int64).to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    
    # from config
    def train_process(
            batches,
            batch_config,
            train_type="train"
        ):
        # accelerator.unwrap_model(transformer).move_to_device_except_swap_blocks(accelerator.device)  # reduce peak memory usage
        # accelerator.unwrap_model(transformer).prepare_block_swap_before_forward()
        # handle batch size > 1
        batch_size = len(batches)
        
        learning_target = None
        noised_latents = None
        prompt_embeds = None
        text_ids = None
        for batch in batches:
            target_config = batch_config["target_config"]
            batch_targets = batch[batch_targets_key][target_config]
            
            batch_references = []
            reference_dropout = 0
            if "reference_config" in batch_config:
                reference_config = batch_config["reference_config"]
                reference_dropout = batch_config["reference_dropout"] if "reference_dropout" in batch_config else 0
                batch_references = batch[batch_references_key][reference_config]
            
            caption_config_key = batch_config["caption_config"]
            batch_caption = batch[batch_captions_key][caption_config_key]
            flush()
            # tcfm_ref_list = []
            target_list = []
            noised_latent_list = []
            reference_list = []
            # dropout_ref_list = []
            # masked_list = []
            # mask_list = []
            # captions = {
                
            # }
            for target in batch_targets:
                # latent_path = target[latent_path_key]
                # latent = get_latent(latent_path)
                latent = torch.stack([target[latent_key]], dim=0)  # (1,16,4,h,w)
                
                latent = Flux2KleinPipeline._patchify_latents(latent)
                latent = latent.to(device=accelerator.device, dtype=weight_dtype)
                latent = (latent - latents_bn_mean) / latents_bn_std
                latent = latent.squeeze(0)
                
                target_list.append(latent)
                if from_latent_path_key in target:
                    # from_latent_path = target[from_latent_path_key]
                    # from_latent = get_latent(from_latent_path)
                    from_latent = target[from_latent_key]
                    noised_latent_list.append(from_latent)
                else:
                    noised_latent_list.append(latent)
            
            for reference in batch_references:
                # latent_path = reference[latent_path_key]
                if reference_dropout > random.random():
                    break
                # latent = get_latent(latent_path)
                latent = torch.stack([reference[latent_key]], dim=0)  # (1,16,4,h,w)
                latent = latent.to(device=accelerator.device, dtype=weight_dtype)
                
                latent = Flux2KleinPipeline._patchify_latents(latent)
                latent = latent.to(device=accelerator.device, dtype=weight_dtype)
                latent = (latent - latents_bn_mean) / latents_bn_std
                latent = latent.to(device=accelerator.device, dtype=weight_dtype)
                latent = latent.squeeze(0)
                reference_list.append(latent)
            
            # cached_npz_path = batch_caption[npz_path_key]
            # prompt_embeds, prompt_embeds_mask = get_caption_embedding(cached_npz_path)
            # prompt_embed, text_id = batch_caption[prompt_embed_key], batch_caption[text_id_key]
            
            prompt_embed = batch_caption[prompt_embed_key]
            
            target_list = torch.stack(target_list, dim=0) # (1,16,1,h,w)
            noised_latent_list = torch.stack(noised_latent_list, dim=0)
            
            
            if prompt_embeds is None:
                prompt_embeds = prompt_embed
            else:
                try:
                    prompt_embeds = torch.cat((prompt_embeds,prompt_embed), dim=0)
                except:
                    print("prompt_embeds cat error size:", prompt_embeds.shape(), prompt_embed.shape())
                    pass
        
            # if text_ids is None:
            #     text_ids = text_id
            # else:
            #     try:
            #         text_ids = torch.cat((text_ids,text_id), dim=0)
            #     except:
            #         print("text_ids cat error size:", text_ids.shape, text_id.shape)
            #         pass
            
            skip_noised_latents = False
            # cat factual_images as image guidance
            if learning_target is None:
                learning_target = target_list
            else:
                try:
                    learning_target = torch.cat((learning_target,target_list), dim=0) # (1,16,1,h,w)
                except:
                    print("learning_target cat error size:", learning_target.shape, target_list.shape)
                    skip_noised_latents = True
                    pass
            
            if skip_noised_latents:
                continue
            if noised_latents is None:
                noised_latents = noised_latent_list
            else:
                try:
                    noised_latents = torch.cat((noised_latents,noised_latent_list), dim=0) # (1,16,1,h,w)
                except:
                    print("noised_latents cat error size:", learning_target.shape, target_list.shape)
                    pass
                
        # batch_size = batch["batch_size"]
        u = compute_density_for_timestep_sampling(
            weighting_scheme=args.weighting_scheme,
            batch_size=batch_size,
            logit_mean=args.logit_mean,
            logit_std=args.logit_std,
            mode_scale=args.mode_scale
        )
        indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
        timesteps = noise_scheduler_copy.timesteps[indices].to(device=accelerator.device)
        
        # noise = torch.randn_like(noised_latents).to(accelerator.device)
        noise = sample_noise(
            target_list=learning_target,
            n_samples=args.nln_samples,
            scale_factor=args.nln_scale,
            method=args.nln_method,
            device=accelerator.device,
            dtype=learning_target.dtype,
        )
        
        # Add noise according to flow matching.
        # zt = (1 - texp) * x + texp * z1
        sigmas = get_sigmas(timesteps, n_dim=noised_latents.ndim, dtype=noised_latents.dtype)
        
        noisy_model_input = (1.0 - sigmas) * noised_latents + sigmas * noise
        
        latents = noisy_model_input
        # pack noisy latents
        # noisy_model_input = noisy_model_input.permute(0, 2, 1, 3, 4)
        model_input = Flux2KleinPipeline._pack_latents(
            noisy_model_input
        )
        model_input_ids = Flux2KleinPipeline._prepare_latent_ids(noised_latents).to(device=model_input.device)
        orig_input_shape = model_input.shape
        orig_input_ids_shape = model_input_ids.shape

        # latent_image_ids = prepare_pos_ids(modality_id=1,
        #                                     type='image',
        #                                     start=(prompt_embeds.shape[1],
        #                                             prompt_embeds.shape[1]),
        #                                     height=latents.shape[2]//2,
        #                                     width=latents.shape[3]//2).to(accelerator.device, dtype=torch.float32)
        
        # # handle partial noised
        # packed_ref_latents = None
        cond_model_input = torch.stack(reference_list, dim=0) if len(reference_list) > 0 else None
        if cond_model_input.shape[0] > 0:
            # cond_model_input_list = [cond_model_input[i].unsqueeze(0) for i in range(cond_model_input.shape[0])]
            ref_latents_ids = Flux2KleinPipeline._prepare_image_ids(reference_list).to(
                device=accelerator.device
            )
            ref_latents_ids = ref_latents_ids.view(
                ref_latents_ids.shape[0], -1, model_input_ids.shape[-1]
            )
            packed_ref_latents = Flux2KleinPipeline._pack_latents(cond_model_input)
            # convert packed_ref_latents from (x, d, c) to (1, x*d, c)
            packed_ref_latents = packed_ref_latents.reshape(model_input.shape[0], -1, packed_ref_latents.shape[-1])
            model_input = torch.cat([model_input, packed_ref_latents], dim=1)
            model_input_ids = torch.cat([model_input_ids, ref_latents_ids], dim=1)
            
        caption_dropout = args.caption_dropout if hasattr(args, "caption_dropout") else 0.0
        # override caption dropout from caption_config
        if "caption_dropout" in batch_config:
            caption_dropout = batch_config["caption_dropout"]
        if caption_dropout > 0 and random.random() < caption_dropout:
            prompt_embed, _ = get_empty_embedding()
            prompt_embeds = prompt_embed.to(device=accelerator.device, dtype=weight_dtype)
            
        text_ids = Flux2KleinPipeline._prepare_text_ids(prompt_embeds)
        text_ids = text_ids.to(accelerator.device)
        
        guidance = None
        with accelerator.autocast():
            # Predict the noise residual
            timesteps = timesteps.expand(latents.shape[0]).to(latents.dtype)
            
            # handle guidance
            if transformer.config.guidance_embeds:
                guidance = torch.full([1], args.guidance_scale, device=accelerator.device)
                guidance = guidance.expand(model_input.shape[0])
            else:
                guidance = None
                
            model_pred = transformer(
                hidden_states=model_input,
                timestep=timesteps / 1000,
                guidance=guidance,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=model_input_ids,
                return_dict=False,
            )[0]
        
        model_pred = model_pred[:, : orig_input_shape[1], :]
        model_input_ids = model_input_ids[:, : orig_input_ids_shape[1], :]

        model_pred = Flux2KleinPipeline._unpack_latents_with_ids(model_pred, model_input_ids)

        # ====================Debug latent====================
        # vae = AutoencoderKL.from_single_file(
        #     vae_path
        # )
        # vae.to(device=accelerator.device)
        # image_processor = VaeImageProcessor(vae_scale_factor=vae.config.scaling_factor)
        # with torch.no_grad():
        #     image = vae.decode(model_pred / vae.config.scaling_factor, return_dict=False)[0]
        # image = image_processor.postprocess(image, output_type="pil")[0]
        # image.save("model_pred.png")
        # ====================Debug latent====================
        
        target = noise - learning_target
        
        weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
        
        # Compute regular loss.
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        ).mean()
        
        # Clean up intermediate variables to reduce memory usage
        del model_pred, target, noise, learning_target, noised_latents
        del noisy_model_input, model_input
        del prompt_embeds, text_ids
        if 'packed_ref_latents' in locals():
            del packed_ref_latents
        # Clean up reference_list if it exists
        if 'reference_list' in locals():
            del reference_list
        
        total_loss = loss
        return total_loss
    
    print("\nStart training...")
    # accelerator.free_memory()  # Critical for HF components
    # torch.cuda.synchronize()  # Ensure all GPU ops complete
    flush()
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            with accelerator.accumulate(transformer):
                dataset_name = batch[0]["dataset"]
                batch_config = get_dataset_batch_config(dataset_name, dataset_configs)
                loss = train_process(
                    batch,
                    batch_config=batch_config
                )
                
                # Backpropagate
                accelerator.backward(loss)
                step_loss = accelerator.reduce(loss.detach(), reduction="mean").item()
                if accelerator.sync_gradients:
                    params_to_clip = transformer_lora_parameters
                    accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)

                # Clean up gradients and perform optimization
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()  # Clear gradients after optimization
                
                # Checks if the accelerator has performed an optimization step behind the scenes
                #post batch check for gradient updates
                # accelerator.wait_for_everyone()
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                
                lr = lr_scheduler.get_last_lr()[0]
                lr_name = "lr"
                if args.optimizer == "prodigy":
                    if resume_step>0 and resume_step == global_step:
                        lr = 0
                    else:
                        lr = lr_scheduler.optimizers[-1].param_groups[0]["d"] * lr_scheduler.optimizers[-1].param_groups[0]["lr"]
                    lr_name = "lr/d*lr"
                logs = {"step_loss": step_loss, lr_name: lr}
                accelerator.log(logs, step=global_step)
                progress_bar.set_postfix(**logs)
                
                # Explicitly delete intermediate variables to free memory
                del loss
                
                if global_step >= max_train_steps:
                    break
                # del step_loss
                flush()
                
                if global_step % args.save_model_steps == 0 and args.save_model_steps > 0:
                    # accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"{args.save_name}-{epoch}-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                
                if global_step % args.save_model_steps == 0 and args.save_model_steps > 0 and os.path.exists(val_metadata_path):
                    # store rng before validation
                    before_state = torch.random.get_rng_state()
                    np_seed = abs(int(args.seed)) if args.seed is not None else np.random.seed()
                    py_state = python_get_rng_state()
                    with torch.no_grad():
                        transformer = unwrap_model(transformer)
                        # freeze rng
                        np.random.seed(val_seed)
                        torch.manual_seed(val_seed)
                        dataloader_generator = torch.Generator()
                        dataloader_generator.manual_seed(val_seed)
                        torch.backends.cudnn.deterministic = True
                        
                        validation_datarows = []
                        with open(val_metadata_path, "r", encoding='utf-8') as readfile:
                            validation_datarows = json.loads(readfile.read(), strict=False)
                        
                        if len(validation_datarows)>0:
                            validation_dataset = CachedJsonDataset(validation_datarows, 
                                                    latents_bn_mean, latents_bn_std, accelerator.device, weight_dtype,
                                                    latent_path_key, latent_key, npz_path_key, prompt_embed_key, prompt_embeds_mask_key, prompt_embed_length_key,
                                                    from_latent_path_key, from_latent_key)
                            # batch_size  = 1
                            val_batch_sampler = BucketBatchSampler(validation_dataset, batch_size=args.train_batch_size)
                            # val_batch_sampler = DistributedBucketBatchSampler(validation_dataset, 
                            #                                         batch_size=args.train_batch_size,
                            #                                         num_replicas=accelerator.num_processes,
                            #                                         rank=accelerator.process_index,
                            #                                         seed=args.seed)
                            val_dataloader = torch.utils.data.DataLoader(
                                validation_dataset,
                                batch_sampler=val_batch_sampler,
                                collate_fn=collate_fn,
                                num_workers=dataloader_num_workers,
                            )
                            
                            # val_dataloader = accelerator.prepare(
                            #     val_dataloader
                            # )

                            print("\nStart val_loss\n")
                            total_loss = 0.0
                            num_batches = len(val_dataloader)
                            # if no val data, skip the following 
                            if num_batches == 0:
                                print("No validation data, skip validation.")
                            else:
                                # basically the as same as the training loop
                                enumerate_val_dataloader = enumerate(val_dataloader)
                                for i, batch in tqdm(enumerate_val_dataloader,position=1):
                                    dataset_name = batch[0]["dataset"]
                                    batch_config = get_dataset_batch_config(dataset_name, dataset_configs)
                                    loss = train_process(
                                        batch,
                                        batch_config=batch_config,
                                        train_type="val"
                                    )
                                    global_loss = accelerator.reduce(loss.detach(), reduction="mean").item()
                                    total_loss+=global_loss
                                    
                                    # Clean up validation step variables to free memory
                                    del loss
                                    
                                accelerator.wait_for_everyone()
                                avg_loss = total_loss / num_batches
                                # convert to float
                                # avg_loss = float(avg_loss.cpu().numpy())
                                
                                # if accelerator.is_main_process:
                                logs = {"val_loss": avg_loss, "epoch": epoch}
                                # print(logs)
                                progress_bar.set_postfix(**logs)
                                accelerator.log(logs, step=global_step)
                                accelerator.wait_for_everyone()
                                
                            flush()
                            print("\nEnd val_loss\n")
                        
                        
                    # restore rng before validation
                    np.random.seed(np_seed)
                    torch.random.set_rng_state(before_state)
                    torch.backends.cudnn.deterministic = False
                    version, state, gauss = py_state
                    python_set_rng_state((version, tuple(state), gauss))
            
                    # del before_state, np_seed, py_state
                    flush()
            
                accelerator.wait_for_everyone()
                            
        # ==================================================
        # validation part
        # ==================================================
        
        if global_step < args.skip_step:
            continue
        
    
        # store rng before validation
        before_state = torch.random.get_rng_state()
        np_seed = abs(int(args.seed)) if args.seed is not None else np.random.seed()
        py_state = python_get_rng_state()
        
        if (epoch >= args.skip_epoch and epoch % args.save_model_epochs == 0) or epoch == args.num_train_epochs - 1 or (global_step % args.save_model_steps == 0 and args.save_model_steps > 0):
            if accelerator.is_main_process:
                save_path = os.path.join(args.output_dir, f"{args.save_name}-{epoch}-{global_step}")
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")
                
        
        # only execute when val_metadata_path exists
        if ((epoch >= args.skip_epoch and epoch % args.validation_epochs == 0) or epoch == args.num_train_epochs - 1 or (global_step % args.save_model_steps == 0 and args.save_model_steps > 0)) and os.path.exists(val_metadata_path):
            with torch.no_grad():
                transformer = unwrap_model(transformer)
                # freeze rng
                np.random.seed(val_seed)
                torch.manual_seed(val_seed)
                dataloader_generator = torch.Generator()
                dataloader_generator.manual_seed(val_seed)
                torch.backends.cudnn.deterministic = True
                
                validation_datarows = []
                with open(val_metadata_path, "r", encoding='utf-8') as readfile:
                    validation_datarows = json.loads(readfile.read(), strict=False)
                
                if len(validation_datarows)>0:
                    validation_dataset = CachedJsonDataset(validation_datarows, 
                                            latents_bn_mean, latents_bn_std, accelerator.device, weight_dtype,
                                            latent_path_key, latent_key, npz_path_key, prompt_embed_key, prompt_embeds_mask_key, prompt_embed_length_key,
                                            from_latent_path_key, from_latent_key)
                    
                    val_batch_sampler = BucketBatchSampler(validation_dataset, batch_size=args.train_batch_size)
                    # val_batch_sampler = DistributedBucketBatchSampler(validation_dataset, 
                    #                                         batch_size=args.train_batch_size,
                    #                                         num_replicas=accelerator.num_processes,
                    #                                         rank=accelerator.process_index,
                    #                                         seed=args.seed)

                    #initialize the DataLoader with the bucket batch sampler
                    val_dataloader = torch.utils.data.DataLoader(
                        validation_dataset,
                        batch_sampler=val_batch_sampler,
                        collate_fn=collate_fn,
                        num_workers=dataloader_num_workers,
                    )
                    # val_dataloader = accelerator.prepare(
                    #     val_dataloader
                    # )

                    print("\nStart val_loss\n")
                    
                    total_loss = 0.0
                    num_batches = len(val_dataloader)
                    # if no val data, skip the following 
                    if num_batches == 0:
                        print("No validation data, skip validation.")
                    else:
                        # basically the as same as the training loop
                        enumerate_val_dataloader = enumerate(val_dataloader)
                        for i, batch in tqdm(enumerate_val_dataloader,position=1):
                            dataset_name = batch[0]["dataset"]
                            batch_config = get_dataset_batch_config(dataset_name, dataset_configs)
                            loss = train_process(
                                batch,
                                batch_config=batch_config,
                                train_type="val"
                            )
                            global_loss = accelerator.reduce(loss.detach(), reduction="mean").item()
                            total_loss+=global_loss
                            
                            # Clean up validation step variables to free memory
                            del loss
                        
                        accelerator.wait_for_everyone()
                        avg_loss = total_loss / num_batches
                        # convert to float
                        # avg_loss = float(avg_loss.cpu().numpy())
                        
                        # if accelerator.is_main_process:
                            
                        lr = lr_scheduler.get_last_lr()[0]
                        logs = {"val_loss": avg_loss, "epoch": epoch}
                        # print(logs)
                        progress_bar.set_postfix(**logs)
                        
                        accelerator.log(logs, step=global_step)
                        accelerator.wait_for_everyone()
                    flush()
                    print("\nEnd val_loss\n")

        # restore rng before validation
        np.random.seed(np_seed)
        torch.random.set_rng_state(before_state)
        torch.backends.cudnn.deterministic = False
        version, state, gauss = py_state
        python_set_rng_state((version, tuple(state), gauss))
        
        # del before_state, np_seed, py_state
        gc.collect()
        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
            
        
        # ==================================================
        # end validation part
        # ==================================================
    
    # Properly close the progress bar to avoid cleanup errors
    if 'progress_bar' in locals():
        progress_bar.close()
    
    accelerator.end_training()
    print("Saved to ")
    print(args.output_dir)
    print_end_signal()
    
    
if __name__ == "__main__": 
    args, config_args = parse_args()
    main(args, config_args)