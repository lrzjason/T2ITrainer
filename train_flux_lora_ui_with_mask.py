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
    FluxPipeline,
    # FluxTransformer2DModel,
)


from flux.transformer_flux_masked import MaskedFluxTransformer2DModel

from flux.flux_utils import compute_loss_weighting_for_sd3, compute_density_for_timestep_sampling

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

# from diffusers import StableDiffusionXLPipeline
# from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256 import StableDiffusionXLPipeline
from tqdm import tqdm 
# from PIL import Image 

from sklearn.model_selection import train_test_split

import json


# import sys
# from utils.image_utils_kolors import BucketBatchSampler, CachedImageDataset, create_metadata_cache
from utils.image_utils_flux import BucketBatchSampler, CachedMaskedPairsDataset

# from prodigyopt import Prodigy


# https://github.com/Lightning-AI/pytorch-lightning/blob/0d52f4577310b5a1624bed4d23d49e37fb05af9e/src/lightning_fabric/utilities/seed.py
from random import getstate as python_get_rng_state
from random import setstate as python_set_rng_state

from peft import LoraConfig, prepare_model_for_kbit_training
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
# from kolors.models.modeling_chatglm import ChatGLMModel
# from kolors.models.tokenization_chatglm import ChatGLMTokenizer

from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

if is_wandb_available():
    import wandb
    
from safetensors.torch import save_file

from utils.dist_utils import flush

from hashlib import md5
import glob
import shutil
from collections import defaultdict

from utils.image_utils_flux import create_empty_embedding, create_embedding, cache_file, cache_multiple


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
# =========Debias implementation from: https://github.com/kohya-ss/sd-scripts/blob/main/library/custom_train_functions.py#L99
# def prepare_scheduler_for_custom_training(noise_scheduler, device):
#     if hasattr(noise_scheduler, "all_snr"):
#         return

#     alphas_cumprod = noise_scheduler.alphas_cumprod
#     sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
#     sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
#     alpha = sqrt_alphas_cumprod
#     sigma = sqrt_one_minus_alphas_cumprod
#     all_snr = (alpha / sigma) ** 2

#     noise_scheduler.all_snr = all_snr.to(device)


# def apply_snr_weight(loss, timesteps, noise_scheduler, gamma, v_prediction=False):
#     snr = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])
#     min_snr_gamma = torch.minimum(snr, torch.full_like(snr, gamma))
#     if v_prediction:
#         snr_weight = torch.div(min_snr_gamma, snr + 1).float().to(loss.device)
#     else:
#         snr_weight = torch.div(min_snr_gamma, snr).float().to(loss.device)
#     loss = loss * snr_weight
#     return loss

# def apply_debiased_estimation(loss, timesteps, noise_scheduler):
#     snr_t = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])  # batch_size
#     snr_t = torch.minimum(snr_t, torch.ones_like(snr_t) * 1000)  # if timestep is 0, snr_t is inf, so limit it to 1000
#     weight = 1 / torch.sqrt(snr_t)
#     loss = weight * loss
#     return loss
# =========Debias implementation from: https://github.com/kohya-ss/sd-scripts/blob/main/library/custom_train_functions.py#L99


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
        default=1100,
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
        default=0.7,
        help="As regularization of objective transfer learning. Set as 1 if you aren't training different objective.",
    )
    parser.add_argument(
        "--reg_timestep",
        type=int,
        default=900,
        help="As regularization of objective transfer learning. You could try different value.",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="locon",
        help="LoRA algorithm to use (locon, loha, lokr, lora)",
    )
    parser.add_argument(
        "--conv_dim",
        type=int,
        default=16,
        help="Convolutional LoRA dimension",
    )
    parser.add_argument(
        "--conv_alpha",
        type=float,
        default=0.5,
        help="Convolutional LoRA alpha",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a couple of iterations then exit",
    )
    
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    # env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    # if env_local_rank != -1 and env_local_rank != args.local_rank:
    #     args.local_rank = env_local_rank

    # if args.with_prior_preservation:
    #     if args.class_data_dir is None:
    #         raise ValueError("You must specify a data directory for class images.")
    #     if args.class_prompt is None:
    #         raise ValueError("You must specify prompt for class images.")
    # else:
    #     # logger is not available yet
    #     if args.class_data_dir is not None:
    #         warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
    #     if args.class_prompt is not None:
    #         warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

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
    
    
    # lr_num_cycles = args.cosine_restarts
    lr_power = 1
    
    # this is for consistence validation. all validation would use this seed to generate the same validation set
    # val_seed = random.randint(1, 100)
    val_seed = 42
    
    # test max_time_steps
    # args.max_time_steps = 600
    
    # args.seed = 4321
    # args.logging_dir = 'logs'
    # args.mixed_precision = "bf16"
    # args.report_to = "wandb"
    
    # args.output_dir = 'F:/models/flux/objectRemoval'
    # args.rank = 32
    # args.skip_epoch = 0
    # args.break_epoch = 0
    # args.skip_step = 0
    # args.gradient_checkpointing = True
    # args.validation_ratio = 0.1
    # args.num_validation_images = 1
    # # args.pretrained_model_name_or_path = "F:/Kolors"
    # args.pretrained_model_name_or_path = "F:/T2ITrainer/flux_models/fill"
    # args.model_path = ""
    # # args.model_path = "F:/models/unet/flux1-dev-fp8-e4m3fn.safetensors"
    # # args.use_fp8 = True
    # args.resume_from_checkpoint = None
    # # args.train_data_dir = "F:/ImageSet/ObjectRemoval/test/I-210618_I01001_W01"
    # # args.train_data_dir = "F:/ImageSet/ObjectRemoval/Subject200K/images/f_padded"
    # # F:\ImageSet\fashion_product_image_dataset\combined_test\Boys
    # args.train_data_dir = "F:/ImageSet/ObjectRemoval/object_removal_alpha"
    # args.resume_from_checkpoint = "F:/models/flux/objectRemoval/objectRemovalBeta_AlphaRegR10-720"
    # args.learning_rate = 1e-4
    # args.optimizer = "adamw"
    # args.lr_warmup_steps = 1
    # args.lr_scheduler = "constant"
    # args.save_model_epochs = 1
    # args.validation_epochs = 1
    # args.train_batch_size = 1
    # args.repeats = 4
    # args.gradient_accumulation_steps = 1
    # args.num_train_epochs = 6
    # args.caption_dropout = 0.1
    # args.mask_dropout = 0.05 
    # args.allow_tf32 = True
    # args.blocks_to_swap = 10
    # args.resolution = 512
    
    # args.reg_ratio = 0.7
    # # args.vae_path = "F:/models/VAE/sdxl_vae.safetensors"

    # args.save_name = "objectRemovalBeta_AlphaRegR10"
    # args.recreate_cache = True
    # args.weighting_scheme = "logit_snr"
    # args.logit_mean = -6.0
    # args.logit_std = 2.0
    
    # stage two 0 inpaint
    # reg_inpaint_ratio = 0
    # stage two 1 inpaint
    reg_inpaint_ratio = 0
    
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
    # noise_scheduler = DDPMScheduler.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="scheduler"
    # )
    
    
    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    
    # prepare noise scheduler
    # noise_scheduler = DDPMScheduler(
    #     beta_start=0.00085, beta_end=0.014, beta_schedule="scaled_linear", num_train_timesteps=1100, clip_sample=False, 
    #     dynamic_thresholding_ratio=0.995, prediction_type="epsilon", steps_offset=1, timestep_spacing="leading", trained_betas=None
    # )
    # if args.use_debias or (args.snr_gamma is not None and args.snr_gamma > 0):
    #     prepare_scheduler_for_custom_training(noise_scheduler, accelerator.device)
    
    
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
            # "ff.net.0.proj",
            # "ff.net.2",
            # "ff_context.net.0.proj",
            # "ff_context.net.2",
        ]
    
    # Make sure the trainable params are in float32.
    # if args.mixed_precision == "fp16":
    #     models = [transformer]
    #     if args.train_text_encoder:
    #         models.extend([text_encoder_one])
    #     # only upcast trainable parameters (LoRA) into fp32
    #     cast_training_params(models, dtype=torch.float32)
        # Define suffixes and base name extraction function
    if args.train_data_dir is not None:
        input_dir = args.train_data_dir
        recreate_cache = args.recreate_cache

        supported_image_types = ['.jpg','.jpeg','.png','.webp']
        files = glob.glob(f"{input_dir}/**", recursive=True)
        image_files = [f for f in files if os.path.splitext(f)[-1].lower() in supported_image_types]

        full_datarows = []
        gt_image_suffix = "_G"
        factual_image_suffix = "_F"
        factual_image_mask_suffix = "_M"
        gt_files = []
        factual_image_files = []
        factual_image_masks = []
        factual_pairs = []
        
        def find_index_from_right(lst, value):
            try:
                reversed_index = lst[::-1].index(value[::-1])
                return len(lst) - 1 - reversed_index
            except:
                return -1
            
        
        # filter images with gt_image_suffix
        for f in image_files:
            base_name = os.path.basename(f)
            filename, _ = os.path.splitext(base_name)
            
            gt_index = find_index_from_right(filename,gt_image_suffix)
            factual_index = find_index_from_right(filename,factual_image_suffix)
            mask_index = find_index_from_right(filename,factual_image_mask_suffix)
            if gt_index > 0:
                gt_files.append(f)
            elif mask_index > 0 and mask_index > factual_index:
                factual_image_masks.append(f)
            elif factual_index > 0:
                factual_image_files.append(f)
                
        # Create a mapping from base filename (without suffix) to ground truth file
        
        gt_mapping = {}
        for gt_file in gt_files:
            base_name = os.path.basename(gt_file)
            filename, _ = os.path.splitext(base_name)
            
            suffix_index = find_index_from_right(filename,gt_image_suffix)
            filename_without_suffix = filename[:suffix_index]
            
            subdir = os.path.dirname(gt_file)
            mapping_key = f"{subdir}_{filename_without_suffix}"  # Remove '_G'
            gt_mapping[mapping_key] = gt_file
             
        # Create a mapping from base filename to mask file
        mask_mapping = {}
        for mask_file in factual_image_masks:
            base_name = os.path.basename(mask_file)
            filename, _ = os.path.splitext(base_name)
            
            mask_index = find_index_from_right(filename,factual_image_mask_suffix)
            filename_without_suffix = filename[:mask_index]
            
            factual_index = find_index_from_right(filename_without_suffix,factual_image_suffix)
            
            if factual_index > 0:
                filename_without_suffix = filename[:factual_index]
            
            subdir = os.path.dirname(mask_file)
            mapping_key = f"{subdir}_{filename_without_suffix}"  # Remove '_G'
            # base_filename = filename[:-len(factual_image_mask_suffix)]  # Remove '_M'
            mask_mapping[mapping_key] = mask_file
            
        for factual_file in factual_image_files:
            base_name = os.path.basename(factual_file)
            filename, _ = os.path.splitext(base_name)
            # base_filename = filename[:-len(factual_image_suffix)]  # Remove '_F'
            
            suffix_index = find_index_from_right(filename,factual_image_suffix)
            filename_without_suffix = filename[:suffix_index]
            
            subdir = os.path.dirname(factual_file)
            mapping_key = f"{subdir}_{filename_without_suffix}"  # Remove '_G'
            
            # Find the corresponding ground truth image based on base_filename
            if mapping_key in gt_mapping:
                gt_file = gt_mapping[mapping_key]
                
                # Find the corresponding mask
                if mapping_key in mask_mapping:
                    mask_file = mask_mapping[mapping_key]
                    
                    # Pair them together
                    factual_pairs.append((gt_file, factual_file, mask_file))
        
        if len(factual_pairs) > 0:
            if os.path.exists(metadata_path) and os.path.exists(val_metadata_path):
                with open(metadata_path, "r", encoding='utf-8') as readfile:
                    metadata_datarows = json.loads(readfile.read())
                    full_datarows += metadata_datarows
                    
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
                
                create_empty_embedding(tokenizers,text_encoders)
                embedding_objects = []
                resolutions = [args.resolution]
                # exist_npz_path = ""
                for gt_file,factual_image_file,factual_image_mask in tqdm(factual_pairs):
                # for image_file in tqdm(image_files):
                    file_name = os.path.basename(factual_image_file)
                    folder_path = os.path.dirname(factual_image_file)
                    
                    # create text embedding based on factual_image
                    f_json = create_embedding( tokenizers,text_encoders,folder_path,file_name,
                        recreate_cache=recreate_cache,
                        # exist_npz_path=exist_npz_path,
                        resolutions=resolutions,
                        )
                    # if use_same_embedding and exist_npz_path != "":
                    #     exist_npz_path = f_json["npz_path"]
                    f_json["ground_true_path"] = gt_file
                    f_json["factual_image_path"] = factual_image_file
                    f_json["factual_image_mask_path"] = factual_image_mask
                    embedding_objects.append(f_json)
                
                # move glm to cpu to reduce vram memory
                # text_encoders[0].to("cpu")
                del text_encoders,tokenizers
                flush()
                metadata_datarows = []
                # cache latent
                print("Cache latent")
                for json_obj in tqdm(embedding_objects):
                    full_obj = cache_multiple(vae,json_obj,recreate_cache=recreate_cache, resolution=resolution)
                    metadata_datarows.append(full_obj)
                    
                full_datarows += metadata_datarows
                
                text_encoder_one.to("cpu")
                text_encoder_two.to("cpu")
                del vae, tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two
            

            # Handle validation split
            if args.validation_ratio > 0:
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
            else:
                datarows = metadata_datarows

            # Save updated metadata
            with open(metadata_path, "w", encoding='utf-8') as outfile:
                outfile.write(json.dumps(datarows))

    flush()
    
    # repeat_datarows = []
    # for datarow in datarows:
    #     for i in range(args.repeats):
    #         repeat_datarows.append(datarow)
    # datarows = repeat_datarows
    
    datarows = datarows * args.repeats
    
    
    
    offload_device = accelerator.device
        
    if not (args.model_path is None or args.model_path == ""):
        config = f"{args.pretrained_model_name_or_path}/transformer/config.json"
        transformer = MaskedFluxTransformer2DModel.from_single_file(args.model_path, 
                            config=config,  
                            torch_dtype=weight_dtype
                        ).to(offload_device)
    else:
        if args.pretrained_model_name_or_path == "black-forest-labs/FLUX.1-dev":
            transformer = MaskedFluxTransformer2DModel.from_pretrained(
                args.pretrained_model_name_or_path, 
                subfolder="transformer",  
                torch_dtype=weight_dtype
            ).to(offload_device)
            flush()
        else:
            # load from repo
            transformer_folder = os.path.join(args.pretrained_model_name_or_path, "transformer")
            # weight_file = "diffusion_pytorch_model"
            variant = None
            transformer = MaskedFluxTransformer2DModel.from_pretrained(
                        transformer_folder, variant=variant,  
                        torch_dtype=weight_dtype
                    ).to(offload_device)
        
            flush()

    transformer = prepare_model_for_kbit_training(transformer, use_gradient_checkpointing=False)
    
    is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0
    if is_swapping_blocks:
        # Swap blocks between CPU and GPU to reduce memory usage, in forward and backward passes.
        logger.info(f"enable block swap: blocks_to_swap={args.blocks_to_swap}")
        transformer.enable_block_swap(args.blocks_to_swap, accelerator.device)


    transformer.requires_grad_(False)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # now we will add new LoRA weights to the attention layers
    if args.algo.lower() in ["locon", "loha", "lokr"]:
        import lycoris.kohya as lyco

        lyco_network = lyco.create_network(
            1.0,
            args.rank,
            args.rank,
            None,
            None,
            transformer,
            algo=args.algo,
            conv_dim=args.conv_dim,
            conv_alpha=args.conv_alpha,
        )
        lyco_network.apply_to()
    else:
        transformer_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
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
            name_split = name.split(".")
            layer_order = name_split[1]
            if int(layer_order) in freezed_layers:
                param.requires_grad = False
        # freeze final layers, suggested by dev (lora not used, it might used in full fine tune)
        # if "norm_out" in name:
        #     param.requires_grad = False
        # if "proj_out" in name:
        #     param.requires_grad = False
    # print(layer_names)
    # layer_names = []
    # for name, param in transformer.named_parameters():
    #     print(f"name: {name} requires_grad:{param.requires_grad}")
    #     layer_names.append(name)
    # print("debug")
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
            FluxPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save
            )
            
            last_part = os.path.basename(os.path.normpath(output_dir))
            file_path = f"{output_dir}/{last_part}.safetensors"
            ori_file = f"{output_dir}/pytorch_lora_weights.safetensors"
            if os.path.exists(ori_file): 
                # copy ori to new name
                shutil.copy(ori_file, file_path)
            
            # # save to kohya
            # peft_state_dict = convert_all_state_dict_to_peft(transformer_lora_layers_to_save)
            # kohya_state_dict = convert_state_dict_to_kohya(peft_state_dict)
            # # add prefix to keys
            # prefix = 'lora_unet_'
            # prefixed_state_dict = {prefix + key: value for key, value in kohya_state_dict.items()}
            # last_part = os.path.basename(os.path.normpath(output_dir))
            # file_path = f"{output_dir}/{last_part}.safetensors"
            # # save comfyui/webui lora as the name of parent
            # save_file(prefixed_state_dict, file_path)

    def load_model_hook(models, input_dir):
        transformer_ = None
        while len(models) > 0:
            model = models.pop()
            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict = FluxPipeline.lora_state_dict(input_dir)
        # lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)

        # unet_state_dict = {f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")}
        # unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
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
        # if args.mixed_precision == "fp16" or args.mixed_precision == "bf16":
        #     models = [transformer_]
        #     # only upcast trainable parameters (LoRA) into fp32
        #     cast_training_params(models)


    if args.algo.lower() not in ["locon", "loha", "lokr"]:
        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True


    
    # resume from cpu after cache files
    # transformer.to(accelerator.device, dtype=weight_dtype)

    # Make sure the trainable params are in float32.
    # if args.mixed_precision == "fp16":
    #     models = [transformer]
    #     # only upcast trainable parameters (LoRA) into fp32
    #     cast_training_params(models, dtype=torch.float32)

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
        # not sure if this would have issue when using multiple aspect ratio
        # latents = torch.stack([example["latent"] for example in examples])
        # time_ids = torch.stack([example["time_id"] for example in examples])
        prompt_embeds = torch.stack([example["prompt_embed"] for example in examples])
        pooled_prompt_embeds = torch.stack([example["pooled_prompt_embed"] for example in examples])
        txt_attention_masks = torch.stack([example["txt_attention_mask"] for example in examples])
        
        sample = {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "txt_attention_masks": txt_attention_masks,
        }
        
        image_classes = ["ground_true", "factual_image", "factual_image_mask", "factual_image_masked_image"]
        for image_class in image_classes:
            sample[image_class] = torch.stack([example[image_class]["latent"] for example in examples])
        return sample
    # create dataset based on input_dir
    train_dataset = CachedMaskedPairsDataset(datarows,conditional_dropout_percent=args.caption_dropout)

    # referenced from everyDream discord minienglish1 shared script
    #create bucket batch sampler
    bucket_batch_sampler = BucketBatchSampler(train_dataset, batch_size=args.train_batch_size, drop_last=True)

    #initialize the DataLoader with the bucket batch sampler
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=bucket_batch_sampler, #use bucket_batch_sampler instead of shuffle
        collate_fn=collate_fn,
        num_workers=dataloader_num_workers,
    )
    
    

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if max_train_steps is None:
        max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
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
    
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            with accelerator.accumulate(transformer):
                
                accelerator.unwrap_model(transformer).move_to_device_except_swap_blocks(accelerator.device)  # reduce peak memory usage
                accelerator.unwrap_model(transformer).prepare_block_swap_before_forward()
                flush()
                
                # latents = batch["latents"].to(accelerator.device)
                prompt_embeds = batch["prompt_embeds"].to(accelerator.device)
                pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(accelerator.device)
                txt_attention_masks = batch["txt_attention_masks"].to(accelerator.device)
                
                ground_trues = batch["ground_true"].to(accelerator.device)
                factual_images = batch["factual_image"].to(accelerator.device)
                factual_image_masks = batch["factual_image_mask"].to(accelerator.device)
                factual_image_masked_images = batch["factual_image_masked_image"].to(accelerator.device)
                
                # random select factual_images and ground_trues
                # ground_trues is a reg selection to prevent model degradation
                # when latents set to factual_images which means the model learning objective is selected to learn to remove objects
                r = random.random()
                latents = factual_images
                # is_inpaint = False
                if r < args.reg_ratio:
                    latents = ground_trues
                    prompt_embeds = torch.zeros_like(prompt_embeds).to(accelerator.device)
                    pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds).to(accelerator.device)
                    
                # latents = ground_trues
                latents = (latents - vae_config_shift_factor) * vae_config_scaling_factor
                latents = latents.to(dtype=weight_dtype)
                                
                # scale ground trues with vae factor
                ground_trues = (ground_trues - vae_config_shift_factor) * vae_config_scaling_factor
                ground_trues = ground_trues.to(dtype=weight_dtype)
                
                factual_image_masked_images = (factual_image_masked_images - vae_config_shift_factor) * vae_config_scaling_factor
                factual_image_masked_images = factual_image_masked_images.to(dtype=weight_dtype)
                
                # text_ids = batch["text_ids"].to(accelerator.device)
                
                text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=accelerator.device, dtype=weight_dtype)
                

                vae_scale_factor = 2 ** (len(vae_config_block_out_channels) - 1)
                # print("vae_scale_factor")
                # print(vae_scale_factor)

                latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                    latents.shape[0],
                    latents.shape[2] // 2,
                    latents.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )
                
                # noise = torch.randn_like(latents)
                noise = torch.randn_like(latents) + args.noise_offset * torch.randn(latents.shape[0], latents.shape[1], 1, 1).to(accelerator.device)
                
                bsz = latents.shape[0]
                
                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=accelerator.device)
                
                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise
                
                # pack noisy latents
                packed_noisy_latents = FluxPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=latents.shape[0],
                    num_channels_latents=latents.shape[1],
                    height=latents.shape[2],
                    width=latents.shape[3],
                )
                
                # implement mask dropout
                if args.mask_dropout > random.random():
                    factual_image_masks = torch.ones_like(factual_image_masks)
                
                # pack factual_image
                packed_factual_image_masks = FluxPipeline._pack_latents(
                    factual_image_masks,
                    batch_size=latents.shape[0],
                    num_channels_latents=vae_scale_factor * vae_scale_factor,
                    height=latents.shape[2],
                    width=latents.shape[3],
                )
                
                # pack factual_image
                packed_factual_image_masked_images = FluxPipeline._pack_latents(
                    factual_image_masked_images,
                    batch_size=latents.shape[0],
                    num_channels_latents=latents.shape[1],
                    height=latents.shape[2],
                    width=latents.shape[3],
                )
                
                masked_image_latents = torch.cat((packed_factual_image_masked_images, packed_factual_image_masks), dim=-1)
                # print("masked_image_latents.shape")
                # print(masked_image_latents.shape)
                # concat noisy latents and masked image latents
                cat_model_input = torch.cat((packed_noisy_latents, masked_image_latents), dim=2)
                # print("cat_model_input.shape")
                # print(cat_model_input.shape)
                
                if handle_guidance:
                    guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                    guidance = guidance.expand(latents.shape[0])
                else:
                    guidance = None
                
                with accelerator.autocast():
                    # Predict the noise residual
                    model_pred = transformer(
                        hidden_states=cat_model_input,
                        encoder_hidden_states=prompt_embeds,
                        joint_attention_kwargs = {'attention_mask': txt_attention_masks},
                        # txt_attention_masks=txt_attention_masks,
                        pooled_projections=pooled_prompt_embeds,
                        # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
                        timestep=timesteps / 1000,
                        img_ids=latent_image_ids,
                        txt_ids=text_ids,
                        guidance=guidance,
                        return_dict=False
                    )[0]
                
                
                model_pred = FluxPipeline._unpack_latents(
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
                
                # learning forward to ground true
                # training the model to predict the velocity of noise - ground_trues
                # model predicted ~= noise - ground_trues
                
                
                # # added reg_inpaint_ratio for prevent inpaint functionality
                # # while training with empty object image could prevent the model degradation 
                # # but the model seems to forget how to paint objects
                # # so we added reg_inpaint_ratio to prevent inpaint functionality
                # if is_inpaint:
                #     # when is_inpaint is true, learning towards to factual image from factual image
                #     # when is_inpaint is false, learning towards to ground_trues from factual image or grounc_trues.
                #     target = noise - latents
                # else:
                #     
                target = noise - ground_trues
                
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                
                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                
                loss = loss.mean()

                # Backpropagate
                accelerator.backward(loss)
                step_loss = loss.detach().item()
                del loss, latents, target, model_pred,  timesteps,  bsz, noise, noisy_model_input
                if accelerator.sync_gradients:
                    params_to_clip = transformer_lora_parameters
                    accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)

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
                    if args.dry_run and global_step >= 2:
                        break
                
                lr = lr_scheduler.get_last_lr()[0]
                lr_name = "lr"
                if args.optimizer == "prodigy":
                    if resume_step>0 and resume_step == global_step:
                        lr = 0
                    else:
                        lr = lr_scheduler.optimizers[-1].param_groups[0]["d"] * lr_scheduler.optimizers[-1].param_groups[0]["lr"]
                    lr_name = "lr/d*lr"
                logs = {"step_loss": step_loss, lr_name: lr, "epoch": epoch}
                accelerator.log(logs, step=global_step)
                progress_bar.set_postfix(**logs)
                
                if global_step >= max_train_steps:
                    break
                if args.dry_run and global_step >= 2:
                    break
                del step_loss
                gc.collect()
                torch.cuda.empty_cache()
                
                
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
                            validation_dataset = CachedMaskedPairsDataset(validation_datarows,conditional_dropout_percent=0)
                            
                            batch_size  = 1
                            # batch_size = args.train_batch_size
                            # handle batch size > validation dataset size
                            # if batch_size > len(validation_datarows):
                            #     batch_size = 1
                            
                            val_batch_sampler = BucketBatchSampler(validation_dataset, batch_size=batch_size, drop_last=True)

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
                                    accelerator.unwrap_model(transformer).move_to_device_except_swap_blocks(accelerator.device)  # reduce peak memory usage
                                    accelerator.unwrap_model(transformer).prepare_block_swap_before_forward()
                                    flush()
                                    
                                    # latents = batch["latents"].to(accelerator.device)
                                    prompt_embeds = batch["prompt_embeds"].to(accelerator.device)
                                    pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(accelerator.device)
                                    txt_attention_masks = batch["txt_attention_masks"].to(accelerator.device)
                                    # text_ids = batch["text_ids"].to(accelerator.device)
                                    ground_trues = batch["ground_true"].to(accelerator.device)
                                    factual_images = batch["factual_image"].to(accelerator.device)
                                    factual_image_masks = batch["factual_image_mask"].to(accelerator.device)
                                    factual_image_masked_images = batch["factual_image_masked_image"].to(accelerator.device)
                                    
                                    factual_images = (factual_images - vae_config_shift_factor) * vae_config_scaling_factor
                                    factual_images = factual_images.to(dtype=weight_dtype)

                                    # scale ground trues with vae factor
                                    ground_trues = (ground_trues - vae_config_shift_factor) * vae_config_scaling_factor
                                    ground_trues = ground_trues.to(dtype=weight_dtype)
                                    
                                    latents = factual_images
                                    if random.random() < args.reg_ratio:
                                        prompt_embeds = torch.zeros_like(prompt_embeds).to(accelerator.device)
                                        pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds).to(accelerator.device)
                                        latents = ground_trues
                                    
                                    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=accelerator.device, dtype=weight_dtype)
                                    
                                    vae_scale_factor = 2 ** (len(vae_config_block_out_channels) - 1)

                                    latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                                        latents.shape[0],
                                        latents.shape[2] // 2,
                                        latents.shape[3] // 2,
                                        accelerator.device,
                                        weight_dtype,
                                    )
                                    
                                    noise = torch.randn_like(latents) + args.noise_offset * torch.randn(latents.shape[0], latents.shape[1], 1, 1).to(accelerator.device)
                                    bsz = latents.shape[0]
                                    
                                    # Sample a random timestep for each image
                                    # for weighting schemes where we sample timesteps non-uniformly
                                    u = compute_density_for_timestep_sampling(
                                        weighting_scheme=args.weighting_scheme,
                                        batch_size=bsz,
                                        logit_mean=args.logit_mean,
                                        logit_std=args.logit_std,
                                        mode_scale=args.mode_scale,
                                    )
                                    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                                    timesteps = noise_scheduler_copy.timesteps[indices].to(device=accelerator.device)
                                    
                                    # Add noise according to flow matching.
                                    # zt = (1 - texp) * x + texp * z1
                                    sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                                    noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise
                                    
                                    # pack noisy latents
                                    packed_noisy_latents = FluxPipeline._pack_latents(
                                        noisy_model_input,
                                        batch_size=latents.shape[0],
                                        num_channels_latents=latents.shape[1],
                                        height=latents.shape[2],
                                        width=latents.shape[3],
                                    )
                                    # pack factual_image
                                    packed_factual_image_masks = FluxPipeline._pack_latents(
                                        factual_image_masks,
                                        batch_size=latents.shape[0],
                                        num_channels_latents=vae_scale_factor * vae_scale_factor,
                                        height=latents.shape[2],
                                        width=latents.shape[3],
                                    )
                                    # pack factual_image
                                    packed_factual_image_masked_images = FluxPipeline._pack_latents(
                                        factual_image_masked_images,
                                        batch_size=latents.shape[0],
                                        num_channels_latents=latents.shape[1],
                                        height=latents.shape[2],
                                        width=latents.shape[3],
                                    )
                                    # print("packed_factual_image_masked_images.shape")
                                    # print(packed_factual_image_masked_images.shape)
                                    # print("packed_factual_image_masks.shape")
                                    # print(packed_factual_image_masks.shape)
                                    masked_image_latents = torch.cat((packed_factual_image_masked_images, packed_factual_image_masks), dim=-1)
                                    # print("masked_image_latents.shape")
                                    # print(masked_image_latents.shape)
                                    # concat noisy latents and masked image latents
                                    cat_model_input = torch.cat((packed_noisy_latents, masked_image_latents), dim=2)
                                    # print("cat_model_input.shape")
                                    # print(cat_model_input.shape)
                                    
                                    
                                    if handle_guidance:
                                        guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                                        guidance = guidance.expand(latents.shape[0])
                                    else:
                                        guidance = None
                                    
                                    with accelerator.autocast():
                                        # Predict the noise residual
                                        model_pred = transformer(
                                            hidden_states=cat_model_input,
                                            # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
                                            timestep=timesteps / 1000,
                                            guidance=guidance,
                                            pooled_projections=pooled_prompt_embeds,
                                            encoder_hidden_states=prompt_embeds,
                                            txt_ids=text_ids,
                                            img_ids=latent_image_ids,
                                            return_dict=False,
                                            joint_attention_kwargs = {'attention_mask': txt_attention_masks},
                                        )[0]
                                    
                                    
                                    model_pred = FluxPipeline._unpack_latents(
                                        model_pred,
                                        height=latents.shape[2] * vae_scale_factor,
                                        width=latents.shape[3] * vae_scale_factor,
                                        vae_scale_factor=vae_scale_factor,
                                    )

                                    # these weighting schemes use a uniform timestep sampling
                                    # and instead post-weight the loss
                                    weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

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
                                    
                                    
                                    # flow matching loss
                                    # if args.precondition_outputs:
                                    #     target = latents
                                    # else:
                                    # target = noise - latents
                                    
                                    # learning forward to ground true
                                    # training the model to predict the velocity of noise - ground_trues
                                    # model predicted ~= noise - ground_trues
                                    target = noise - ground_trues
                                    
                                    weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                                    
                                    # Compute regular loss.
                                    loss = torch.mean(
                                        (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                                        1,
                                    )
                                    loss = loss.mean()

                                    total_loss+=loss.detach()
                                    del latents, target, loss, model_pred,  timesteps,  bsz, noise, packed_noisy_latents
                                    gc.collect()
                                    torch.cuda.empty_cache()
                                    
                                avg_loss = total_loss / num_batches
                                
                                lr = lr_scheduler.get_last_lr()[0]
                                lr_name = "val_lr"
                                if args.optimizer == "prodigy":
                                    lr = lr_scheduler.optimizers[-1].param_groups[0]["d"] * lr_scheduler.optimizers[-1].param_groups[0]["lr"]
                                    lr_name = "val_lr lr/d*lr"
                                logs = {"val_loss": avg_loss, lr_name: lr, "epoch": epoch}
                                print(logs)
                                progress_bar.set_postfix(**logs)
                                accelerator.log(logs, step=global_step)
                                del num_batches, avg_loss, total_loss
                            del validation_datarows, validation_dataset, val_batch_sampler, val_dataloader
                            gc.collect()
                            torch.cuda.empty_cache()
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
                    validation_dataset = CachedMaskedPairsDataset(validation_datarows,conditional_dropout_percent=0)
                    
                    batch_size  = 1
                    # batch_size = args.train_batch_size
                    # handle batch size > validation dataset size
                    # if batch_size > len(validation_datarows):
                    #     batch_size = 1
                    
                    val_batch_sampler = BucketBatchSampler(validation_dataset, batch_size=batch_size, drop_last=True)

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
                            accelerator.unwrap_model(transformer).move_to_device_except_swap_blocks(accelerator.device)  # reduce peak memory usage
                            accelerator.unwrap_model(transformer).prepare_block_swap_before_forward()
                            flush()
                            
                            # latents = batch["latents"].to(accelerator.device)
                            prompt_embeds = batch["prompt_embeds"].to(accelerator.device)
                            pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(accelerator.device)
                            txt_attention_masks = batch["txt_attention_masks"].to(accelerator.device)
                            # text_ids = batch["text_ids"].to(accelerator.device)
                            ground_trues = batch["ground_true"].to(accelerator.device)
                            factual_images = batch["factual_image"].to(accelerator.device)
                            factual_image_masks = batch["factual_image_mask"].to(accelerator.device)
                            factual_image_masked_images = batch["factual_image_masked_image"].to(accelerator.device)
                            
                            
                            latents = factual_images
                            
                            # scale ground trues with vae factor
                            ground_trues = (ground_trues - vae_config_shift_factor) * vae_config_scaling_factor
                            ground_trues = ground_trues.to(dtype=weight_dtype)
                            
                            
                            text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=accelerator.device, dtype=weight_dtype)
                            
                            latents = (latents - vae_config_shift_factor) * vae_config_scaling_factor
                            latents = latents.to(dtype=weight_dtype)

                            vae_scale_factor = 2 ** (len(vae_config_block_out_channels) - 1)

                            latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                                latents.shape[0],
                                latents.shape[2] // 2,
                                latents.shape[3] // 2,
                                accelerator.device,
                                weight_dtype,
                            )
                            
                            noise = torch.randn_like(latents) + args.noise_offset * torch.randn(latents.shape[0], latents.shape[1], 1, 1).to(accelerator.device)
                            bsz = latents.shape[0]
                            # Sample a random timestep for each image
                            # for weighting schemes where we sample timesteps non-uniformly
                            u = compute_density_for_timestep_sampling(
                                weighting_scheme=args.weighting_scheme,
                                batch_size=bsz,
                                logit_mean=args.logit_mean,
                                logit_std=args.logit_std,
                                mode_scale=args.mode_scale,
                            )
                            indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                            timesteps = noise_scheduler_copy.timesteps[indices].to(device=accelerator.device)
                            
                            # Add noise according to flow matching.
                            # zt = (1 - texp) * x + texp * z1
                            sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                            noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise
                            
                            # pack noisy latents
                            packed_noisy_latents = FluxPipeline._pack_latents(
                                noisy_model_input,
                                batch_size=latents.shape[0],
                                num_channels_latents=latents.shape[1],
                                height=latents.shape[2],
                                width=latents.shape[3],
                            )
                            
                            # pack factual_image
                            packed_factual_image_masks = FluxPipeline._pack_latents(
                                factual_image_masks,
                                batch_size=latents.shape[0],
                                num_channels_latents=vae_scale_factor * vae_scale_factor,
                                height=latents.shape[2],
                                width=latents.shape[3],
                            )
                            # pack factual_image
                            packed_factual_image_masked_images = FluxPipeline._pack_latents(
                                factual_image_masked_images,
                                batch_size=latents.shape[0],
                                num_channels_latents=latents.shape[1],
                                height=latents.shape[2],
                                width=latents.shape[3],
                            )
                            # print("packed_factual_image_masked_images.shape")
                            # print(packed_factual_image_masked_images.shape)
                            # print("packed_factual_image_masks.shape")
                            # print(packed_factual_image_masks.shape)
                            masked_image_latents = torch.cat((packed_factual_image_masked_images, packed_factual_image_masks), dim=-1)
                            # print("masked_image_latents.shape")
                            # print(masked_image_latents.shape)
                            # concat noisy latents and masked image latents
                            cat_model_input = torch.cat((packed_noisy_latents, masked_image_latents), dim=2)
                            # print("cat_model_input.shape")
                            # print(cat_model_input.shape)
                            
                            
                            if handle_guidance:
                                guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                                guidance = guidance.expand(latents.shape[0])
                            else:
                                guidance = None
                            
                            with accelerator.autocast():
                                # Predict the noise residual
                                model_pred = transformer(
                                    hidden_states=cat_model_input,
                                    # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
                                    timestep=timesteps / 1000,
                                    guidance=guidance,
                                    pooled_projections=pooled_prompt_embeds,
                                    encoder_hidden_states=prompt_embeds,
                                    txt_ids=text_ids,
                                    img_ids=latent_image_ids,
                                    return_dict=False,
                                    joint_attention_kwargs = {'attention_mask': txt_attention_masks},
                                )[0]
                            
                            
                            model_pred = FluxPipeline._unpack_latents(
                                model_pred,
                                height=latents.shape[2] * vae_scale_factor,
                                width=latents.shape[3] * vae_scale_factor,
                                vae_scale_factor=vae_scale_factor,
                            )

                            # these weighting schemes use a uniform timestep sampling
                            # and instead post-weight the loss
                            weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

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
                            
                            
                            # flow matching loss
                            # if args.precondition_outputs:
                            #     target = latents
                            # else:
                            # target = noise - latents
                            
                            # learning forward to ground true
                            # training the model to predict the velocity of noise - ground_trues
                            # model predicted ~= noise - ground_trues
                            target = noise - ground_trues
                            
                            weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                            
                            # Compute regular loss.
                            loss = torch.mean(
                                (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                                1,
                            )
                            loss = loss.mean()

                            total_loss+=loss.detach()
                            del latents, target, loss, model_pred,  timesteps,  bsz, noise, packed_noisy_latents
                            gc.collect()
                            torch.cuda.empty_cache()
                            
                        avg_loss = total_loss / num_batches
                        
                        lr = lr_scheduler.get_last_lr()[0]
                        lr_name = "val_lr"
                        if args.optimizer == "prodigy":
                            lr = lr_scheduler.optimizers[-1].param_groups[0]["d"] * lr_scheduler.optimizers[-1].param_groups[0]["lr"]
                            lr_name = "val_lr lr/d*lr"
                        logs = {"val_loss": avg_loss, lr_name: lr, "epoch": epoch}
                        print(logs)
                        progress_bar.set_postfix(**logs)
                        accelerator.log(logs, step=global_step)
                        del num_batches, avg_loss, total_loss
                    del validation_datarows, validation_dataset, val_batch_sampler, val_dataloader
                    gc.collect()
                    torch.cuda.empty_cache()
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