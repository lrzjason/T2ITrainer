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
from diffusers.models.model_loading_utils import load_model_dict_into_meta
# import jsonlines

import copy
import safetensors
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
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import diffusers

# from diffusers.image_processor import VaeImageProcessor

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
# from datasets import load_dataset
# from packaging import version
# from torchvision import transforms
# from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
# from transformers import AutoTokenizer, PretrainedConfig
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    # FluxTransformer2DModel,
)

from flux.transformer_flux_masked import MaskedFluxTransformer2DModel
from flux.flux_utils import compute_loss_weighting_for_sd3, compute_density_for_timestep_sampling
from flux.pipeline_flux_kontext import FluxKontextPipeline

from pathlib import Path
from diffusers.optimization import get_scheduler
# from diffusers.training_utils import _set_state_dict_into_text_encoder, cast_training_params, compute_snr
from diffusers.training_utils import (
    cast_training_params,
    compute_snr
)
from diffusers.utils import (
    # check_min_version,
    convert_all_state_dict_to_peft,
    convert_state_dict_to_diffusers,
    convert_state_dict_to_kohya,
    convert_unet_state_dict_to_peft,
    # compute_density_for_timestep_sampling,
    is_wandb_available,
    # compute_loss_weighting_for_sd3,
)
from diffusers.loaders import LoraLoaderMixin
# from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from tqdm import tqdm 
# from PIL import Image 

from sklearn.model_selection import train_test_split

from pathlib import Path
import json


# import sys
# from utils.image_utils_kolors import BucketBatchSampler, CachedImageDataset, create_metadata_cache
from utils.image_utils_flux import CachedMutiImageDataset
from utils.bucket.bucket_batch_sampler import BucketBatchSampler

# from prodigyopt import Prodigy


# https://github.com/Lightning-AI/pytorch-lightning/blob/0d52f4577310b5a1624bed4d23d49e37fb05af9e/src/lightning_fabric/utilities/seed.py
from random import getstate as python_get_rng_state
from random import setstate as python_set_rng_state

from peft import LoraConfig, prepare_model_for_kbit_training
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
# from kolors.models.modeling_chatglm import ChatGLMModel
# from kolors.models.tokenization_chatglm import ChatGLMTokenizer

from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast, BitsAndBytesConfig

if is_wandb_available():
    import wandb
    
from safetensors.torch import save_file

from utils.dist_utils import flush

from hashlib import md5
import glob
import shutil
from collections import defaultdict

from utils.image_utils_flux import load_image, compute_text_embeddings, replace_non_utf8_characters, create_empty_embedding, get_empty_embedding, cache_file, cache_multiple, crop_image,get_md5_by_path,vae_encode,read_image


# from diffusers import FluxPriorReduxPipeline
import cv2

from torchvision import transforms

from diffusers.image_processor import VaeImageProcessor

from utils.utils import find_index_from_right, ToTensorUniversal




def load_text_encoders(class_one, class_two):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2"
    )
    # text_encoder_three = class_three.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="text_encoder_3"
    # )
    return text_encoder_one, text_encoder_two #, text_encoder_three


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


logger = get_logger(__name__)

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
        default=None,
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
                if hasattr(args, key):
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
                else:
                    print(f"Config file contains unknown argument: '{key}'. Ignoring.")
        except Exception as e:
            print(f"Could not load config file '{args.config_path}': {e}. Using command-line arguments.")

    print(f"Using config: {args}")
    return args

def main(args):
    
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
    
    # not use
    reg_timestep = args.reg_timestep
    reg_ratio = args.reg_ratio
    
    
    transformer_subfolder = "transformer"
    
    # enable_redux_training = False
    image_1 = "train"
    image_2 = "ref"
    
    embbeding_path_key = "npz_path"
    embbeding_md5 = f"{embbeding_path_key}_md5"
    caption_key = "captions"
    prompt_embed_key = "prompt_embed"
    pooled_prompt_embed_key = f"pooled_{prompt_embed_key}"
    txt_attention_mask_key = "txt_attention_mask"
    image_path_key = "image_path"
    latent_key = "latent"
    latent_path_key = f"{latent_key}_path"
    latent_md5 = f'{latent_path_key}_md5'
    cache_ext=".npflux"
    latent_ext=".npfluxlatent"
    dataset_based_image = image_1
    image_configs = {
        image_1:{
            "suffix":"_T"
        },
        image_2:{
            "suffix":"_R"
        },
    }
    merge_configs = {**image_configs}
    exclude_base_image_keys = [key for key in merge_configs.keys() if key != dataset_based_image]
    caption_configs = {
        image_1:{
            "ext":"txt",
            # "redux":[image_1,image_2]
        }
    }
    training_layout_configs = {
        image_1: {
            "target": image_1,
            "noised": True,
        },
        # generation image on left and refs on right
        image_2: {
            "target": image_2,
        },
    }
    
    captions_selection = {
        "target": image_1,
        # "use_extra" : True,
        # "condition_extra": {
        #     image_2: 0.5,
        #     "dropout": 0.5,
        # },
        "dropout": args.caption_dropout,
    }
    dataset_configs = {
        "caption_key":caption_key,
        "latent_key":"latent",
        "latent_path_key":latent_path_key,
        "npz_path_key": embbeding_path_key,
        "npz_keys": {
            prompt_embed_key:prompt_embed_key,
            pooled_prompt_embed_key:pooled_prompt_embed_key,
            txt_attention_mask_key:txt_attention_mask_key
        },
    }
    
    # to avoid cache mutiple times on same embedding
    # use_same_embedding = True
    
    lr_num_cycles = args.cosine_restarts
    resolution = int(args.resolution)
    
    
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    if not os.path.exists(args.logging_dir): os.makedirs(args.logging_dir)
    
    
    # create metadata.jsonl if not exist
    metadata_suffix = "flux"
    metadata_path = os.path.join(args.train_data_dir, f'metadata_{metadata_suffix}.json')
    val_metadata_path =  os.path.join(args.train_data_dir, f'val_metadata_{metadata_suffix}.json')
    
    logging_dir = "logs"
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # run name is save name + datetime
    run_name = f"{args.save_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    
    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        target_modules = [
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ]
    
    if args.train_data_dir is not None:
        input_dir = args.train_data_dir
        recreate_cache = args.recreate_cache

        SUPPORTED_IMAGE_TYPES = ['.jpg','.jpeg','.png','.webp']
        # files = glob.glob(f"{input_dir}/**", recursive=True)
        # image_files = [f for f in files if os.path.splitext(f)[-1].lower() in supported_image_types]
        image_files = [
            f for f in glob.iglob(os.path.join(input_dir, "**"), recursive=True)
            if os.path.isfile(f)
            and os.path.splitext(f)[1].lower() in SUPPORTED_IMAGE_TYPES
        ]

        full_datarows = []
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
                suffix = config["suffix"]
                suffix_index = find_index_from_right(filename,suffix)
                if suffix_index > 0:
                    image_pool[key].append(f)
                    
        # construct image pairs
        for key in merge_configs.keys():
            config = merge_configs[key]
            imageset = image_pool[key]
            suffix = config["suffix"]
            prefix = ""
            if "prefix" in config:
                prefix = config["prefix"]
                
            for file in imageset:
                base_name = os.path.basename(file)
                filename, _ = os.path.splitext(base_name)
                
                suffix_index = find_index_from_right(filename,suffix)
                filename_without_suffix = filename[:suffix_index]
                
                # handle prefix setting like mask which related to image
                if "prefix" in config:
                    prefix = config["prefix"]
                    prefix_index = find_index_from_right(filename_without_suffix,prefix)
                    if prefix_index > 0:
                        filename_without_suffix = filename[:prefix_index]
                    
                subdir = os.path.dirname(file)
                mapping_key = f"{subdir}_{filename_without_suffix}"
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
                have_pair = True
                for image_group_key in exclude_base_image_keys:
                    if mapping_key in mapping[image_group_key]:
                        if len(mapping[image_group_key][mapping_key]) > 1:
                            for pair_image in mapping[image_group_key][mapping_key]:
                                if filename in pair_image:
                                    pair[image_group_key] = pair_image
                        else:
                            pair[image_group_key] = mapping[image_group_key][mapping_key][0]
                    # if any image doesn't have fullset pair, it should be skipped
                    else:
                        have_pair = False
                if have_pair:
                    image_pairs.append(pair)
        
        
        if len(image_pairs) > 0:
            with torch.no_grad():
                if os.path.exists(metadata_path) and not args.recreate_cache:
                    with open(metadata_path, "r", encoding='utf-8') as readfile:
                        metadata_datarows = json.loads(readfile.read())
                        full_datarows += metadata_datarows
                    
                    if os.path.exists(val_metadata_path):
                        with open(val_metadata_path, "r", encoding='utf-8') as readfile:
                            val_metadata_datarows = json.loads(readfile.read())
                            full_datarows += val_metadata_datarows
                else:
                    # Offload models to CPU and load necessary components
                    tokenizer_one = CLIPTokenizer.from_pretrained(
                        args.pretrained_model_name_or_path,
                        subfolder="tokenizer",
                    )
                    tokenizer_two = T5TokenizerFast.from_pretrained(
                        args.pretrained_model_name_or_path,
                        subfolder="tokenizer_2",
                    )

                    # import correct text encoder classes
                    text_encoder_cls_one = import_model_class_from_model_name_or_path(
                        args.pretrained_model_name_or_path, 
                    )
                    text_encoder_cls_two = import_model_class_from_model_name_or_path(
                        args.pretrained_model_name_or_path,  subfolder="text_encoder_2"
                    )
                    text_encoder_one, text_encoder_two = load_text_encoders(
                        text_encoder_cls_one, text_encoder_cls_two
                    )
                    
                    vae = AutoencoderKL.from_pretrained(
                        args.pretrained_model_name_or_path,
                        subfolder="vae",
                    )
                    
                    vae.requires_grad_(False)
                    text_encoder_one.requires_grad_(False)
                    text_encoder_two.requires_grad_(False)
                    
                    vae.to(accelerator.device, dtype=torch.float32)
                    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
                    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
                    tokenizers = [tokenizer_one,tokenizer_two]
                    text_encoders = [text_encoder_one,text_encoder_two]
                    
                    # repo_redux = "black-forest-labs/FLUX.1-Redux-dev"
                    # pipe_prior_redux = None
                    # if enable_redux_training:
                    #     pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(repo_redux, torch_dtype=weight_dtype).to(accelerator.device)
                    
                    create_empty_embedding(tokenizers,text_encoders)
                    embedding_objects = {}
                    resolutions = [args.resolution]
                    
                    for image_pair in tqdm(image_pairs):
                        embedding_object = {
                            caption_key:{}
                        }
                        mapping_key = image_pair["mapping_key"]
                        for caption_config_key in caption_configs.keys():
                            caption_config = caption_configs[caption_config_key]
                            # redux_image_path = image_pair[dataset_based_image]
                            image_file = image_pair[dataset_based_image]
                            
                                
                            filename = os.path.basename(image_file)
                            folder_path = os.path.dirname(image_file)
                            # get filename and ext from file
                            filename, _ = os.path.splitext(filename)
                            image_path = image_file
                            json_obj = {
                            }
                            # read caption
                            caption_ext = '.txt'
                            text_path = os.path.join(folder_path, f'{filename}{caption_ext}')
                            content = ''

                            file_path = os.path.join(folder_path, filename)
                            npz_path = f'{file_path}{cache_ext}'
                            json_obj["npz_path"] = npz_path
                            
                            if os.path.exists(text_path):
                                json_obj["text_path"] = text_path
                                try:
                                    content = open(text_path, encoding='utf-8').read()
                                    # json_obj['prompt'] = content
                                    json_obj["text_path_md5"] = get_md5_by_path(text_path)
                                except:
                                    content = open(text_path, encoding='utf-8').read()
                                    # try to remove non utf8 character
                                    content = replace_non_utf8_characters(content)
                                    # json_obj['prompt'] = content
                                    json_obj["text_path_md5"] = ""
                            
                            if not recreate_cache and os.path.exists(npz_path):
                                if 'npz_path_md5' not in json_obj:
                                    json_obj["npz_path_md5"] = get_md5_by_path(npz_path)
                                npz_dict = torch.load(npz_path)
                            else:
                                prompt_embeds, pooled_prompt_embeds, txt_attention_masks = compute_text_embeddings(text_encoders,tokenizers,content,device=text_encoders[0].device)
                                prompt_embed = prompt_embeds.squeeze(0)
                                pooled_prompt_embed = pooled_prompt_embeds.squeeze(0)
                                txt_attention_mask = txt_attention_masks.squeeze(0)
                                npz_dict = {
                                    "prompt_embed": prompt_embed.cpu(), 
                                    "pooled_prompt_embed": pooled_prompt_embed.cpu(),
                                    "txt_attention_mask": txt_attention_mask.cpu(),
                                }
                                
                                torch.save(npz_dict, npz_path)
                            embedding_object[caption_key][caption_config_key] = json_obj
                            
                        for key in image_pair.keys():
                            if key == "mapping_key":
                                embedding_object["mapping_key"] = image_pair["mapping_key"]
                            else:
                                embedding_object[key] = {
                                    image_path_key:image_pair[key]
                                }
                        embedding_objects[mapping_key] = embedding_object
                    # move glm to cpu to reduce vram memory
                    # text_encoders[0].to("cpu")
                    del text_encoders,tokenizers
                    flush()
                    metadata_datarows = []
                    # cache latent
                    print("Cache latent")
                    for embedding_object_key in tqdm(embedding_objects.keys()):
                        json_obj = embedding_objects[embedding_object_key]
                        train_transforms = transforms.Compose([ToTensorUniversal(), transforms.Normalize([0.5], [0.5])])
                        
                        temp_image_pool = { }
                        for image_config_key in image_configs.keys():
                            image_config = image_configs[image_config_key]
                            image_path = json_obj[image_config_key][image_path_key]
                            filename, _ = os.path.splitext(image_path)
                            
                            json_obj[image_config_key][latent_path_key] = f"{filename}{latent_ext}"
                            latent_cache_path = f"{filename}{latent_ext}"
                            image = crop_image(image_path,resolution=resolution)
                            image_height, image_width, _ = image.shape
                            
                            json_obj['bucket'] = f"{image_width}x{image_height}"
                            
                            # skip if already cached
                            if os.path.exists(latent_cache_path) and not recreate_cache:
                                if latent_md5 not in json_obj[image_config_key]:
                                    json_obj[image_config_key][latent_md5] = get_md5_by_path(latent_cache_path)
                                    for caption_file_key in json_obj[caption_key].keys():
                                        npz_path = json_obj[caption_key][caption_file_key][embbeding_path_key]
                                        json_obj[caption_key][caption_file_key][embbeding_md5] = get_md5_by_path(npz_path)
                                continue
                            
                            image = train_transforms(image)
                                
                            temp_image_pool[image_config_key] = {
                                "image":image,
                                "height":image_height,
                                "width":image_width,
                            }
                            latent_dict = vae_encode(vae,image)
                            torch.save(latent_dict, latent_cache_path)
                            json_obj[image_config_key][latent_md5] = get_md5_by_path(latent_cache_path)
                        # del npz_dict
                        flush()
                        
                        del temp_image_pool
                        metadata_datarows.append(json_obj)
                        
                    full_datarows += metadata_datarows
                    
                    text_encoder_one.to("cpu")
                    text_encoder_two.to("cpu")
                    # del vae, tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two, pipe_prior_redux
                    del vae, tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two
                
            datarows = metadata_datarows
            # Handle validation split
            if args.validation_ratio > 0:
                if not os.path.exists(val_metadata_path):
                    train_ratio = 1 - args.validation_ratio
                    validation_ratio = args.validation_ratio
                    if len(full_datarows) == 1:
                        full_datarows = full_datarows + full_datarows.copy()
                        validation_ratio = 0.5
                        train_ratio = 0.5
                    training_datarows, validation_datarows = train_test_split(
                        full_datarows,
                        train_size=train_ratio,
                        test_size=validation_ratio
                    )
                    datarows = training_datarows
                    # Save validation metadata
                    if len(validation_datarows) > 0:
                        with open(val_metadata_path, "w", encoding='utf-8') as outfile:
                            outfile.write(json.dumps(validation_datarows))
                    # Clear memory
                    del validation_datarows

            # Save updated metadata
            with open(metadata_path, "w", encoding='utf-8') as outfile:
                outfile.write(json.dumps(datarows))


    flush()
    
    datarows = datarows * args.repeats
    
    
    offload_device = accelerator.device
        
    if not (args.model_path is None or args.model_path == ""):
        # config = f"{args.pretrained_model_name_or_path}/transformer/config.json"
        transformer = MaskedFluxTransformer2DModel.from_single_file(args.model_path, 
                            # config=config,  
                            torch_dtype=weight_dtype
                        ).to(offload_device)
    else:
        if args.pretrained_model_name_or_path == "black-forest-labs/FLUX.1-dev":
            transformer = MaskedFluxTransformer2DModel.from_pretrained(
                args.pretrained_model_name_or_path, 
                subfolder=transformer_subfolder,  
                torch_dtype=weight_dtype
            ).to(offload_device)
        else:
            # load from repo
            transformer_folder = os.path.join(args.pretrained_model_name_or_path, transformer_subfolder)
            # weight_file = "diffusion_pytorch_model"
            variant = None
            transformer = MaskedFluxTransformer2DModel.from_pretrained(
                        transformer_folder, variant=variant,  
                        torch_dtype=weight_dtype
                    ).to(offload_device)
    flush()

    if "quantization_config" in transformer.config:
        transformer = prepare_model_for_kbit_training(transformer, use_gradient_checkpointing=False)
    else:
        transformer = transformer.to(offload_device, dtype=weight_dtype)
        transformer.requires_grad_(False)
    
    is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0
    if is_swapping_blocks:
        # Swap blocks between CPU and GPU to reduce memory usage, in forward and backward passes.
        logger.info(f"enable block swap: blocks_to_swap={args.blocks_to_swap}")
        transformer.enable_block_swap(args.blocks_to_swap, accelerator.device)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

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
    layer_names = []
    freezed_layers = []
    if args.freeze_transformer_layers is not None and args.freeze_transformer_layers != '':
        splited_layers = args.freeze_transformer_layers.split()
        for layer in splited_layers:
            layer_name = int(layer.strip())
            freezed_layers.append(layer_name)
    # Freeze the layers
    for name, param in transformer.named_parameters():
        layer_names.append(name)
        if "transformer" in name:
            if '_orig_mod.' in name:
                name = name.replace('_orig_mod.', '')
            name_split = name.split(".")
            layer_order = name_split[1]
            if int(layer_order) in freezed_layers:
                param.requires_grad = False
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder atten layers
            transformer_lora_layers_to_save = None
            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            # save all
            FluxKontextPipeline.save_lora_weights(
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
            configs = {
                "training_layout_configs": training_layout_configs,
                "captions_selection": captions_selection,
                "dataset_configs": dataset_configs,
            }
            configs_file = os.path.join(output_dir, "config_details.json")
            with open(configs_file, "w", encoding='utf-8') as outfile:
                outfile.write(json.dumps(configs, indent=4, ensure_ascii=False))
                
    def load_model_hook(models, input_dir):
        transformer_ = None
        while len(models) > 0:
            model = models.pop()
            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict = FluxKontextPipeline.lora_state_dict(input_dir)
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
        
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

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
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(adam_beta1, adam_beta2),
            beta3=prodigy_beta3,
            d_coef=prodigy_d_coef,
            weight_decay=adam_weight_decay,
            eps=adam_epsilon,
            decouple=prodigy_decouple,
            use_bias_correction=prodigy_use_bias_correction,
            safeguard_warmup=prodigy_safeguard_warmup,
        )
    
    # ================================================================
    # End create embedding 
    # ================================================================
    
    def collate_fn(examples):
        sample = {}
        example = examples[0]
        sample["batch_size"] = len(examples)
        sample["mapping_key"] = [example["mapping_key"] for example in examples]
        for key in example.keys():
            if key == "mapping_key":
                continue
            sample[key] = {}
            sample[key][latent_key] = torch.stack([example[key][latent_key] for example in examples])
            # if masked_latent_key in example[key]:
            #     sample[key][masked_latent_key] = torch.stack([example[key][masked_latent_key] for example in examples])
            if caption_key in example[key]:
                sample[key][caption_key] = {}
                for npz_key in dataset_configs["npz_keys"]:
                    sample[key][caption_key][npz_key] = torch.stack([example[key][caption_key][npz_key] for example in examples])
                # if "redux" in example[key][caption_key]:
                #     sample[key][caption_key]["redux"] = {}
                #     npz_extra_key_groups = dataset_configs["npz_extra_keys"].keys()
                #     for npz_extra_key_group in npz_extra_key_groups:
                #         sample[key][caption_key]["redux"][npz_extra_key_group] = {}
                #         for npz_extra_key in dataset_configs["npz_extra_keys"][npz_extra_key_group].keys():
                #             if npz_extra_key in example[key][caption_key]["redux"][npz_extra_key_group]:
                #                 sample[key][caption_key]["redux"][npz_extra_key_group][npz_extra_key] = torch.stack([example[key][caption_key]["redux"][npz_extra_key_group][npz_extra_key] for example in examples])
                        
        return sample
    # create dataset based on input_dir
    train_dataset = CachedMutiImageDataset(datarows,conditional_dropout_percent=args.caption_dropout, dataset_configs=dataset_configs)

    # referenced from everyDream discord minienglish1 shared script
    #create bucket batch sampler
    bucket_batch_sampler = BucketBatchSampler(train_dataset, batch_size=args.train_batch_size)

    #initialize the DataLoader with the bucket batch sampler
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=bucket_batch_sampler, #use bucket_batch_sampler instead of shuffle
        collate_fn=collate_fn,
        num_workers=dataloader_num_workers,
    )
    
    

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


    # vae config from vae config file
    vae_config_shift_factor = 0.1159
    vae_config_scaling_factor = 0.3611
    vae_config_block_out_channels = [
        128,
        256,
        512,
        512
    ]
    

    print("  Num examples = ", len(train_dataset))
    print("  Num Epochs = ", args.num_train_epochs)
    print("  num_update_steps_per_epoch = ", num_update_steps_per_epoch)
    print("  max_train_steps = ", max_train_steps)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
        num_cycles=lr_num_cycles,
        power=lr_power,
    )
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        optimizer, train_dataloader, lr_scheduler
    )
    
    # load transformer to cpu
    transformer.to("cuda")
    flush()
    
    transformer = accelerator.prepare(transformer, device_placement=[not is_swapping_blocks])
    

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "flux-lora"
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
    global_step = 0
    first_epoch = 0

    resume_step = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint and args.resume_from_checkpoint != "":
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith(args.save_name)]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[-1])

            initial_global_step = global_step
            resume_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

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
                        "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
                    )

                optimizer = optimizer_class(
                    params_to_optimize,
                    lr=args.learning_rate,
                    betas=(adam_beta1, adam_beta2),
                    beta3=prodigy_beta3,
                    d_coef=prodigy_d_coef,
                    weight_decay=adam_weight_decay,
                    eps=adam_epsilon,
                    decouple=prodigy_decouple,
                    use_bias_correction=prodigy_use_bias_correction,
                    safeguard_warmup=prodigy_safeguard_warmup,
                )
            
            lr_scheduler = get_scheduler(
                args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                num_training_steps=max_train_steps * accelerator.num_processes,
                num_cycles=lr_num_cycles,
                power=lr_power,
            )
            
            optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)
    else:
        initial_global_step = 0
    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    
    # max_time_steps = noise_scheduler.config.num_train_timesteps
    # if args.max_time_steps is not None and args.max_time_steps > 0:
    #     max_time_steps = args.max_time_steps
        
                    
    # handle guidance
    if accelerator.unwrap_model(transformer).config.guidance_embeds:
        handle_guidance = True
    else:
        handle_guidance = False
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    def train_process(
            batch,
            training_layout_configs,
            dataset_configs,
            captions_selection,
            # enable_prior_loss=True
        ):
        
        default_sample_size = 128
        
        accelerator.unwrap_model(transformer).move_to_device_except_swap_blocks(accelerator.device)  # reduce peak memory usage
        accelerator.unwrap_model(transformer).prepare_block_swap_before_forward()
        flush()
        batch_size = batch["batch_size"]
        vae_scale_factor = 2 ** (len(vae_config_block_out_channels) - 1)
        u = compute_density_for_timestep_sampling(
            weighting_scheme=args.weighting_scheme,
            batch_size=batch_size,
            logit_mean=args.logit_mean,
            logit_std=args.logit_std,
            mode_scale=args.mode_scale
        )
        indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
        timesteps = noise_scheduler_copy.timesteps[indices].to(device=accelerator.device)
        # factual, groundtrue
        for image_config_key in image_configs.keys():
            # scale latents
            latent = batch[image_config_key][latent_key]
            latent = (latent - vae_config_shift_factor) * vae_config_scaling_factor
            batch[image_config_key][latent_key] = latent.to(device=accelerator.device,dtype=weight_dtype)
        
        latent_list = []
        noised_latent_list = []
        target_list = []
        # masked_list = []
        # mask_list = []
        captions = {
            
        }
        
        for training_layout_config_key in training_layout_configs.keys():
            training_layout_config = training_layout_configs[training_layout_config_key]
            
            # normal image
            x = batch[training_layout_config_key][latent_key]
            
            # caption_key = "captions"
            if caption_key in batch[training_layout_config_key]:
                captions[training_layout_config_key] = {}
                for npz_key in dataset_configs["npz_keys"].keys():
                    captions[training_layout_config_key][npz_key] =  batch[training_layout_config_key][caption_key][npz_key]
                # check extra keys in dataset
                # if "npz_extra_keys" in dataset_configs:
                #     captions[training_layout_config_key]["redux"] = {}
                #     npz_extra_key_groups = dataset_configs["npz_extra_keys"].keys()
                #     for npz_extra_key_group in npz_extra_key_groups:
                #         # check extra keys in batch captions
                #         if npz_extra_key_group in batch[training_layout_config_key][caption_key]["redux"]:
                #             # npz_extra_key_group is redux or something else
                #             npz_extra_keys = dataset_configs["npz_extra_keys"][npz_extra_key_group]
                #             captions[training_layout_config_key]["redux"][npz_extra_key_group] = {}
                #             for npz_extra_key in npz_extra_keys:
                #                 captions[training_layout_config_key]["redux"][npz_extra_key_group][npz_extra_key] = batch[training_layout_config_key][caption_key]["redux"][npz_extra_key_group][npz_extra_key]

            if "transition" in training_layout_config:
                transition_config = training_layout_config["transition"]
                timesteps_config = transition_config["timesteps"]
                from_timestep =  timesteps_config["from"]
                to_timestep =  timesteps_config["to"]
                mask = (from_timestep >= timesteps) & (timesteps >= to_timestep)
                # mask_for_latents = mask.view((-1,) + (1,) * (batch_size - 1))
                mask_for_latents = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                
                images_config = transition_config["images"]
                from_image_key =  images_config["from"]
                from_image = batch[from_image_key][latent_key]
                
                to_image_key =  images_config["to"]
                to_image = batch[to_image_key][latent_key]
                x = torch.where(mask_for_latents, from_image, to_image)
            
            if "target" in training_layout_config:
                target_key = training_layout_config["target"]
                x = batch[target_key][latent_key]
            
            x = x.to(device=accelerator.device,dtype=weight_dtype)
            # control to noise which latent
            if "noised" in training_layout_config and training_layout_config["noised"]:
                noised_latent_list.append(x)
                target_list.append(x)
            else:        
                latent_list.append(x)
                
        noised_latents = torch.cat(noised_latent_list, dim=0)
        noise = torch.randn_like(noised_latents) + args.noise_offset * torch.randn(noised_latents.shape[0], noised_latents.shape[1], 1, 1).to(accelerator.device)
        
        # Add noise according to flow matching.
        # zt = (1 - texp) * x + texp * z1
        sigmas = get_sigmas(timesteps, n_dim=noised_latents.ndim, dtype=noised_latents.dtype)
        noisy_model_input = (1.0 - sigmas) * noised_latents + sigmas * noise
        
        latents = noisy_model_input
        # pack noisy latents
        packed_noisy_latents = FluxKontextPipeline._pack_latents(
            noisy_model_input,
            batch_size=latents.shape[0],
            num_channels_latents=latents.shape[1],
            height=latents.shape[2],
            width=latents.shape[3],
        )
        
        ref_image_ids = None
        packed_ref_latents = None
        # handle partial noised
        if len(latent_list) > 0:
            ref_latents = torch.cat(latent_list, dim=0)   
            # pack noisy latents
            packed_ref_latents = FluxKontextPipeline._pack_latents(
                ref_latents,
                batch_size=ref_latents.shape[0],
                num_channels_latents=ref_latents.shape[1],
                height=ref_latents.shape[2],
                width=ref_latents.shape[3],
            )
            ref_image_ids = FluxKontextPipeline._prepare_latent_image_ids(
                ref_latents.shape[0],
                ref_latents.shape[2] // 2,
                ref_latents.shape[3] // 2,
                accelerator.device,
                weight_dtype,
            )
            ref_image_ids[..., 0] = 1
        
        # cat factual_images as image guidance
        learning_target = torch.cat(target_list, dim=0)
        
        latent_image_ids = FluxKontextPipeline._prepare_latent_image_ids(
            latents.shape[0],
            latents.shape[2] // 2,
            latents.shape[3] // 2,
            accelerator.device,
            weight_dtype,
        )
        
        if ref_image_ids is not None:
            latent_image_ids = torch.cat([latent_image_ids, ref_image_ids], dim=0)  # dim 0 is sequence dimension

        model_input = packed_noisy_latents
        # add ref to channel
        if packed_ref_latents is not None:
            model_input = torch.cat((packed_noisy_latents, packed_ref_latents), dim=1)
            
        if handle_guidance:
            guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None
        
        caption_target = captions_selection["target"]
        # caption_target should always in captions
        assert caption_target in captions
        selected_caption = captions[caption_target]
        
        final_caption = {
            prompt_embed_key : selected_caption[prompt_embed_key],
            pooled_prompt_embed_key : selected_caption[pooled_prompt_embed_key],
            txt_attention_mask_key : selected_caption[txt_attention_mask_key]
        }
        # handle redux
        # if "use_extra" in captions_selection and enable_redux_training:
        #     assert "condition_extra" in captions_selection
        #     conditions = captions_selection["condition_extra"]
            
        #     options = list(conditions.keys())
        #     weights = list(conditions.values())

        #     # Randomly select one option based on weights
        #     condition_selection = random.choices(options, weights=weights, k=1)[0]
            
        #     if condition_selection != "dropout":
        #         final_caption = {}
        #         if "npz_extra_keys" in dataset_configs:
        #             for npz_extra_key in dataset_configs["npz_extra_keys"][condition_selection]:
        #                 final_caption[npz_extra_key] = selected_caption["redux"][condition_selection][npz_extra_key]
        
                
        prompt_embeds = final_caption[prompt_embed_key].to(device=accelerator.device, dtype=weight_dtype)
        pooled_prompt_embeds = final_caption[pooled_prompt_embed_key].to(device=accelerator.device, dtype=weight_dtype)
        
        txt_attention_masks = None
        # if txt_attention_mask_key in final_caption:
        #     txt_attention_masks = final_caption[txt_attention_mask_key].to(device=accelerator.device, dtype=weight_dtype)
            
        if random.random() < captions_selection["dropout"]:
            prompt_embeds = torch.zeros_like(prompt_embeds)
            pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
            # if txt_attention_masks is not None:
            #     txt_attention_masks = torch.zeros_like(txt_attention_masks)
        
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=accelerator.device, dtype=weight_dtype)
        
        with accelerator.autocast():
            # Predict the noise residual
            model_pred = transformer(
                hidden_states=model_input,
                encoder_hidden_states=prompt_embeds,
                # joint_attention_kwargs = {'attention_mask': txt_attention_masks},
                # txt_attention_masks=txt_attention_masks,
                pooled_projections=pooled_prompt_embeds,
                # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
                timestep=timesteps / 1000,
                img_ids=latent_image_ids,
                txt_ids=text_ids,
                guidance=guidance,
                return_dict=False
            )[0]
        
        model_pred = model_pred[:, : packed_noisy_latents.size(1)]
        
        model_pred = FluxKontextPipeline._unpack_latents(
            model_pred,
            height=latents.shape[2] * vae_scale_factor,
            width=latents.shape[3] * vae_scale_factor,
            vae_scale_factor=vae_scale_factor,
        )

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
        _,_,_,t_w = target.shape
        _,_,_,p_w = model_pred.shape
        # split model_pred based on target width
        if p_w != t_w:
            model_pred = model_pred[..., :t_w]
        
        weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
        
        # Compute regular loss.
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        
        loss = loss.mean()
        
        total_loss = loss
        
        return total_loss
            
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            with accelerator.accumulate(transformer):
                loss = train_process(
                    batch,
                    training_layout_configs,
                    dataset_configs,
                    captions_selection,
                    # enable_prior_loss=enable_prior_loss
                    # ,
                    # vae=vae
                )

                # Backpropagate
                accelerator.backward(loss)
                step_loss = loss.detach().item()
                if accelerator.sync_gradients:
                    params_to_clip = transformer_lora_parameters
                    accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)

                del loss
                flush()
                # ensure model in cuda
                transformer.to(accelerator.device)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
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
                            validation_datarows = json.loads(readfile.read())
                        
                        if len(validation_datarows)>0:
                            validation_dataset = CachedMutiImageDataset(validation_datarows,conditional_dropout_percent=args.caption_dropout, dataset_configs=dataset_configs)
                            batch_size  = 1
                            val_batch_sampler = BucketBatchSampler(validation_dataset, batch_size=batch_size)
                            val_dataloader = torch.utils.data.DataLoader(
                                validation_dataset,
                                batch_sampler=val_batch_sampler, #use bucket_batch_sampler instead of shuffle
                                collate_fn=collate_fn,
                                num_workers=dataloader_num_workers,
                            )

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
                                    loss = train_process(
                                        batch,
                                        training_layout_configs,
                                        dataset_configs,
                                        captions_selection,
                                        # enable_prior_loss=enable_prior_loss
                                        # ,
                                        # vae=vae
                                    )
                                    total_loss+=loss.detach()
                                    # del latents, target, loss, model_pred,  timesteps,  bsz, noise, packed_noisy_latents
                                    # flush()
                                    
                                avg_loss = total_loss / num_batches
                                # convert to float
                                avg_loss = float(avg_loss.cpu().numpy())
                                # adaptive_scheduler.update(avg_loss)
                                
                                # lr = lr_scheduler.get_last_lr()[0]
                                # lr_name = "val_lr"
                                # if args.optimizer == "prodigy":
                                #     lr = lr_scheduler.optimizers[-1].param_groups[0]["d"] * lr_scheduler.optimizers[-1].param_groups[0]["lr"]
                                #     lr_name = "val_lr lr/d*lr"
                                logs = {"val_loss": avg_loss, "epoch": epoch}
                                # print(logs)
                                progress_bar.set_postfix(**logs)
                                accelerator.log(logs, step=global_step)
                                
                                
                                # current_logit_mean, current_logit_std = adaptive_scheduler.get_current_params()
                                # accelerator.log({"val_current_logit_mean":current_logit_mean}, step=global_step)
                                # del num_batches, avg_loss, total_loss
                            # del validation_datarows, validation_dataset, val_batch_sampler, val_dataloader
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
            # accelerator.wait_for_everyone()
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
                    validation_datarows = json.loads(readfile.read())
                
                if len(validation_datarows)>0:
                    validation_dataset = CachedMutiImageDataset(validation_datarows,conditional_dropout_percent=args.caption_dropout, dataset_configs=dataset_configs)
                    
                    batch_size  = 1
                    
                    val_batch_sampler = BucketBatchSampler(validation_dataset, batch_size=batch_size)

                    #initialize the DataLoader with the bucket batch sampler
                    val_dataloader = torch.utils.data.DataLoader(
                        validation_dataset,
                        batch_sampler=val_batch_sampler, #use bucket_batch_sampler instead of shuffle
                        collate_fn=collate_fn,
                        num_workers=dataloader_num_workers,
                    )

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
                            loss = train_process(
                                batch,
                                training_layout_configs,
                                dataset_configs,
                                captions_selection,
                                # enable_prior_loss=enable_prior_loss
                                # ,
                                # vae=vae
                            )

                            total_loss+=loss.detach()
                            # del latents, target, loss, model_pred,  timesteps,  bsz, noise, packed_noisy_latents
                            # flush()
                            
                        avg_loss = total_loss / num_batches
                        avg_loss = float(avg_loss.cpu().numpy())
                        # adaptive_scheduler.update(avg_loss)
                        
                        lr = lr_scheduler.get_last_lr()[0]
                        logs = {"val_loss": avg_loss, "epoch": epoch}
                        print(logs)
                        progress_bar.set_postfix(**logs)
                        
                        accelerator.log(logs, step=global_step)
                        
                        
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
        
        
        # ==================================================
        # end validation part
        # ==================================================
    
    accelerator.end_training()
    print("Saved to ")
    print(args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)