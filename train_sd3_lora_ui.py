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
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
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
from utils.image_utils_sd35 import BucketBatchSampler, CachedImageDataset, create_metadata_cache

# from prodigyopt import Prodigy


# https://github.com/Lightning-AI/pytorch-lightning/blob/0d52f4577310b5a1624bed4d23d49e37fb05af9e/src/lightning_fabric/utilities/seed.py
from random import getstate as python_get_rng_state
from random import setstate as python_set_rng_state

from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
# from kolors.models.modeling_chatglm import ChatGLMModel
# from kolors.models.tokenization_chatglm import ChatGLMTokenizer

from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

# try:
#     from diffusers.utils import randn_tensor
# except:
#     from diffusers.utils.torch_utils import randn_tensor

if is_wandb_available():
    import wandb
    
from safetensors.torch import save_file

from utils.dist_utils import flush

from hashlib import md5
import glob
import shutil

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.30.0.dev0")


# https://github.com/huggingface/diffusers/blob/main/src/diffusers/training_utils.py
def compute_loss_weighting_for_sd3(weighting_scheme: str, sigmas=None):
    """
    Computes loss weighting scheme for SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "sigma_sqrt":
        weighting = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weighting = 2 / (math.pi * bot)
    else:
        weighting = torch.ones_like(sigmas)
    return weighting

# https://github.com/huggingface/diffusers/blob/main/src/diffusers/training_utils.py#L236
def compute_density_for_timestep_sampling(
    weighting_scheme: str, batch_size: int, logit_mean: float = None, logit_std: float = None, mode_scale: float = None
):
    """
    Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u

def load_text_encoders(class_one, class_two, class_three):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2"
    )
    text_encoder_three = class_three.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_3"
    )
    return text_encoder_one, text_encoder_two, text_encoder_three


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, variant="fp16"
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
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


def apply_snr_weight(loss, timesteps, noise_scheduler, gamma, v_prediction=False):
    snr = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])
    min_snr_gamma = torch.minimum(snr, torch.full_like(snr, gamma))
    if v_prediction:
        snr_weight = torch.div(min_snr_gamma, snr + 1).float().to(loss.device)
    else:
        snr_weight = torch.div(min_snr_gamma, snr).float().to(loss.device)
    loss = loss * snr_weight
    return loss

def apply_debiased_estimation(loss, timesteps, noise_scheduler):
    snr_t = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])  # batch_size
    snr_t = torch.minimum(snr_t, torch.ones_like(snr_t) * 1000)  # if timestep is 0, snr_t is inf, so limit it to 1000
    weight = 1 / torch.sqrt(snr_t)
    loss = weight * loss
    return loss
# =========Debias implementation from: https://github.com/kohya-ss/sd-scripts/blob/main/library/custom_train_functions.py#L99


def memory_stats():
    print("\nmemory_stats:\n")
    print(torch.cuda.memory_allocated()/1024**2)
    print(torch.cuda.memory_cached()/1024**2)

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
        default="sd3-dreambooth",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
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
        default="sd3_",
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
        choices=["no", "fp16", "bf16"],
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
        "--use_dora",
        action="store_true",
        help="Use dora on peft config",
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
        "--vae_path",
        type=str,
        default=None,
        help=("seperate vae path"),
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default='1024',
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
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"],
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
        "--precondition_outputs",
        type=int,
        default=1,
        help="Flag indicating if we are preconditioning the model outputs or not as done in EDM. This affects how "
        "model `target` is calculated.",
    )
    
    parser.add_argument(
        "--freeze_transformer_layers",
        type=str,
        default='5,7,10,17,18,19',
        help="Stop training the transformer layers included in the input using ',' to seperate layers. Example: 5,7,10,17,18,19"
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
    
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    if not os.path.exists(args.logging_dir): os.makedirs(args.logging_dir)
    
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
    
    
    lr_num_cycles = args.cosine_restarts
    lr_power = 1
    
    # this is for consistence validation. all validation would use this seed to generate the same validation set
    val_seed = random.randint(1, 100)
    
    # test max_time_steps
    # args.max_time_steps = 600
    
    # args.seed = 4321
    # args.logging_dir = 'logs'
    # args.mixed_precision = "bf16"
    # args.report_to = "wandb"
    
    # args.output_dir = 'F:/models/kolors'
    # args.save_name = "kolors_test"
    # args.rank = 32
    # args.skip_epoch = 1
    # args.break_epoch = 0
    # args.skip_step = 0
    # args.gradient_checkpointing = True
    # args.validation_ratio = 0.1
    # args.num_validation_images = 1
    # args.pretrained_model_name_or_path = "F:/Kolors"
    # args.model_path = None # "F:/models/Stable-diffusion/sd3/opensd3.safetensors"
    # # args.resume_from_checkpoint = "F:/models/hy/hy_test-1600"
    # args.resume_from_checkpoint = None
    # # args.resume_from_lora_dir = "F:/models/hy/hy_test-1600"
    # args.train_data_dir = "F:/ImageSet/pixart_test_cropped"
    # # args.train_data_dir = "F:/ImageSet/kolors_test"
    # # args.train_data_dir = "F:/ImageSet/pixart_test_one"
    
    # args.learning_rate = 1e-4
    # args.optimizer = "adamw"
    # args.lr_warmup_steps = 1
    # args.lr_scheduler = "cosine"
    # args.save_model_epochs = 1
    # args.validation_epochs = 1
    # args.train_batch_size = 1
    # args.repeats = 1
    # args.gradient_accumulation_steps = 1
    # args.num_train_epochs = 5
    # args.use_dora = False
    # args.caption_dropout = 0.2
    # args.vae_path = "F:/models/VAE/sdxl_vae.safetensors"

    
    
    # create metadata.jsonl if not exist
    metadata_suffix = "sd35"
    metadata_path = os.path.join(args.train_data_dir, f'metadata_{metadata_suffix}.json')
    val_metadata_path =  os.path.join(args.train_data_dir, f'val_metadata_{metadata_suffix}.json')
    
    logging_dir = "test"
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
    
    offload_device = accelerator.device
    
    if not os.path.exists(metadata_path) or not os.path.exists(val_metadata_path):
        offload_device = torch.device("cpu")
    
    # load from repo
    if args.pretrained_model_name_or_path == "stabilityai/stable-diffusion-3.5-large":
        transformer  = SD3Transformer2DModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="transformer"
            ).to(offload_device, dtype=weight_dtype)
    else:
        # load from repo
        transformer_folder = os.path.join(args.pretrained_model_name_or_path, "transformer")
        # weight_file = "diffusion_pytorch_model"
        variant = None
        # ext = ".safetensors"
        # diffusion_pytorch_model.fp16.safetensors
        # fp16_weight = os.path.join(transformer_folder, f"{weight_file}.fp16{ext}")
        # fp32_weight = os.path.join(transformer_folder, f"{weight_file}{ext}")
        # if os.path.exists(fp16_weight):
        #     variant = "fp16"
        # elif os.path.exists(fp32_weight):
        #     variant = None
        # else:
        #     raise FileExistsError(f"{fp16_weight} and {fp32_weight} not found. \n Please download the model from https://huggingface.co/Kwai-Kolors/Kolors or https://hf-mirror.com/Kwai-Kolors/Kolors")
            
        transformer = SD3Transformer2DModel.from_pretrained(
                    transformer_folder, variant=variant
                ).to(offload_device, dtype=weight_dtype)
    
    if not (args.model_path is None or args.model_path == ""):
        # load from file
        state_dict = safetensors.torch.load_file(args.model_path, device="cpu")
        unexpected_keys = load_model_dict_into_meta(
            transformer,
            state_dict,
            device=offload_device,
            dtype=torch.float32,
            model_name_or_path=args.model_path,
        )
        # updated_state_dict = unet.state_dict()
        if len(unexpected_keys) > 0:
            print(f"Unexpected keys in state_dict: {unexpected_keys}")
        transformer.to(offload_device, dtype=weight_dtype)
        del state_dict,unexpected_keys
        flush()

    transformer.requires_grad_(False)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # now we will add new LoRA weights to the attention layers
    transformer_lora_config = LoraConfig(
        use_dora=args.use_dora,
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
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
            StableDiffusion3Pipeline.save_lora_weights(
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

        lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)

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
        if args.mixed_precision == "fp16" or args.mixed_precision == "bf16":
            models = [transformer_]
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)


    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    

    # ==========================================================
    # Create train dataset
    # ==========================================================
    # data_files = {}
    # this part need more work afterward, you need to prepare 
    # the train files and val files split first before the training
    if args.train_data_dir is not None:
        input_dir = args.train_data_dir
        datarows = []
        cache_list = []
        recreate_cache = args.recreate_cache
        # resolution = args.resolution_config.split(",")
        
        supported_image_types = ['.jpg','.jpeg','.png','.webp']
        files = glob.glob(f"{input_dir}/**", recursive=True)
        image_files = [f for f in files if os.path.splitext(f)[-1].lower() in supported_image_types]
        
        # function to remove metadata datarows which in not exist in directory
        def align_metadata(datarows,image_files,metadata_path):
            new_metadatarows = []
            for metadata_datarow in datarows:
                # if some row not in current image_files, ignore it
                if metadata_datarow['image_path'] in image_files:
                    new_metadatarows.append(metadata_datarow)
            # save new_metadatarows at metadata_path
            with open(metadata_path, "w", encoding='utf-8') as writefile:
                writefile.write(json.dumps(new_metadatarows))
            return new_metadatarows
        
        metadata_datarows = []
        # single_image_training = False
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding='utf-8') as readfile:
                metadata_datarows = json.loads(readfile.read())
                # remove images in metadata_datarows or val_metadata_datarows but not in image_files, handle deleted images
                metadata_datarows = align_metadata(metadata_datarows,image_files,metadata_path)
        # else:
        #     single_image_training = len(image_files) == 1
        
        val_metadata_datarows = []
        if os.path.exists(val_metadata_path):
            with open(val_metadata_path, "r", encoding='utf-8') as readfile:
                val_metadata_datarows = json.loads(readfile.read())
                # remove images in metadata_datarows or val_metadata_datarows but not in image_files, handle deleted images
                val_metadata_datarows = align_metadata(val_metadata_datarows,image_files,val_metadata_path)
        
        # full datarows is aligned, all datarows conatins exists image
        if len(metadata_datarows) == 1:
            full_datarows = metadata_datarows
            # single_image_training = True
        else:
            full_datarows = metadata_datarows + val_metadata_datarows
            
        datarows = full_datarows
        # if not single_image_training:
        #     single_image_training = (len(resolution) > 1 and len(full_datarows) == len(resolution)) or len(full_datarows) == len(resolution)
        # no metadata file, all files should be cached
        cache_list = []
        if (len(datarows) == 0) or recreate_cache:
            cache_list = image_files
        else:
            md5_pairs = [
                {
                    "path":"image_path",
                    "md5": "image_path_md5"
                },
                {
                    "path":"text_path",
                    "md5": "text_path_md5"
                },
                {
                    "path":"npz_path",
                    "md5": "npz_path_md5"
                },
                {
                    "path":"latent_path",
                    "md5": "latent_path_md5"
                },
            ]
            def check_md5(datarows,md5_pairs):
                cache_list = []
                new_datarows = []
                for datarow in tqdm(datarows):
                    corrupted = False
                    # loop all the md5 pairs
                    for pair in md5_pairs:
                        path_name = pair['path']
                        md5_name = pair['md5']
                        # if md5 not in datarow, then recache
                        if not md5_name in datarow.keys():
                            if datarow['image_path'] not in cache_list:
                                cache_list.append(datarow['image_path'])
                                corrupted = True
                            break
                        
                        file_path = datarow[path_name]
                        file_path_md5 = ''
                        if os.path.exists(file_path):
                            with open(file_path, 'rb') as f:
                                file_path_md5 = md5(f.read()).hexdigest()
                        
                        if file_path_md5 != datarow[md5_name]:
                            if datarow['image_path'] not in cache_list:
                                cache_list.append(datarow['image_path'])
                                corrupted = True
                            break
                    if not corrupted:
                        new_datarows.append(datarow)
                return cache_list, new_datarows
                                
            # for metadata_file in metadata_files:
            # Validate two datasets 
            # loop all the datarow and check file md5 for integrity
            # print(f"Checking integrity: ")
            # # fine images not in full_datarows, handle added images
            current_images = [d['image_path'] for d in full_datarows]
            missing_images = [f for f in image_files if f not in current_images]
            if len(missing_images) > 0:
                print(f"Images exists but not in metadata: {len(missing_images)}")
                # add missing images to cache list
                cache_list += missing_images
            
            # check full_datarows md5
            corrupted_files, new_datarows = check_md5(full_datarows,md5_pairs)
            # skip corrupted datarows, update full datarows
            full_datarows = new_datarows
            if len(corrupted_files) > 0:
                print(f"corrupted files: {len(corrupted_files)}")
                # add corrupted files to cache list
                cache_list += corrupted_files
                    
        if len(cache_list)>0:
            # for cpu offload
            transformer.to("cpu")
            # Load the tokenizers
            # tokenizer_one = ChatGLMTokenizer.from_pretrained(
            #     args.pretrained_model_name_or_path,
            #     subfolder="text_encoder",
            #     revision=revision, 
            #     variant=variant
            # )

            # text_encoder_one = ChatGLMModel.from_pretrained(
            #     args.pretrained_model_name_or_path, subfolder="text_encoder", revision=revision, variant=variant
            # )
            
            # Load the tokenizers
            tokenizer_one = CLIPTokenizer.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="tokenizer",
            )
            tokenizer_two = CLIPTokenizer.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="tokenizer_2",
            )
            tokenizer_three = T5TokenizerFast.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="tokenizer_3",
            )

            # import correct text encoder classes
            text_encoder_cls_one = import_model_class_from_model_name_or_path(
                args.pretrained_model_name_or_path, 
            )
            text_encoder_cls_two = import_model_class_from_model_name_or_path(
                args.pretrained_model_name_or_path,  subfolder="text_encoder_2"
            )
            text_encoder_cls_three = import_model_class_from_model_name_or_path(
                args.pretrained_model_name_or_path, subfolder="text_encoder_3"
            )
            
            text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
                text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three
            )
            
            vae = AutoencoderKL.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="vae",
            )
            
            
            # vae_folder = os.path.join(args.pretrained_model_name_or_path, "vae")
            # if args.vae_path:
            #     vae = AutoencoderKL.from_single_file(
            #         args.vae_path,
            #         config=vae_folder,
            #     )
            # else:
            #     # load from repo
            #     weight_file = "diffusion_pytorch_model"
            #     vae_variant = None
            #     ext = ".safetensors"
            #     # diffusion_pytorch_model.fp16.safetensors
            #     fp16_weight = os.path.join(vae_folder, f"{weight_file}.fp16{ext}")
            #     fp32_weight = os.path.join(vae_folder, f"{weight_file}{ext}")
            #     if os.path.exists(fp16_weight):
            #         vae_variant = "fp16"
            #     elif os.path.exists(fp32_weight):
            #         vae_variant = None
            #     else:
            #         raise FileExistsError(f"{fp16_weight} and {fp32_weight} not found. \n Please download the model from https://huggingface.co/Kwai-Kolors/Kolors or https://hf-mirror.com/Kwai-Kolors/Kolors")
                    
            #     vae = AutoencoderKL.from_pretrained(
            #             args.pretrained_model_name_or_path, variant=vae_variant
            #         )
            
            # vae.requires_grad_(False)
            # text_encoder_one.requires_grad_(False)
            vae.requires_grad_(False)
            text_encoder_one.requires_grad_(False)
            text_encoder_two.requires_grad_(False)
            text_encoder_three.requires_grad_(False)
            
            vae.to(accelerator.device, dtype=torch.float32)
            text_encoder_one.to(accelerator.device, dtype=weight_dtype)
            text_encoder_two.to(accelerator.device, dtype=weight_dtype)
            text_encoder_three.to(accelerator.device, dtype=weight_dtype)

            tokenizers = [tokenizer_one,tokenizer_two,tokenizer_three]
            text_encoders = [text_encoder_one,text_encoder_two,text_encoder_three]
            # create metadata and latent cache
            cached_datarows = create_metadata_cache(tokenizers,text_encoders,vae,cache_list,metadata_path=metadata_path,recreate_cache=args.recreate_cache, resolution_config=args.resolution)
            
            # merge newly cached datarows to full_datarows
            full_datarows += cached_datarows
            
            # reset validation_datarows
            validation_datarows = []
            # prepare validation_slipt
            if args.validation_ratio > 0:
                # buckets = image_utils.get_buckets()
                train_ratio = 1 - args.validation_ratio
                validation_ratio = args.validation_ratio
                if len(full_datarows) == 1:
                    full_datarows = full_datarows + full_datarows.copy()
                    validation_ratio = 0.5
                    train_ratio = 0.5
                training_datarows, validation_datarows = train_test_split(full_datarows, train_size=train_ratio, test_size=validation_ratio)
                datarows = training_datarows
            else:
                datarows = full_datarows
            
            # Serializing json
            json_object = json.dumps(datarows, indent=4)
            # update metadata file
            with open(metadata_path, "w", encoding='utf-8') as outfile:
                outfile.write(json_object)
            
            if len(validation_datarows) > 0:
                # Serializing json
                val_json_object = json.dumps(validation_datarows, indent=4)
                # update val metadata file
                with open(val_metadata_path, "w", encoding='utf-8') as outfile:
                    outfile.write(val_json_object)
                
            # clear memory
            del validation_datarows
            text_encoder_one.to("cpu")
            text_encoder_two.to("cpu")
            text_encoder_three.to("cpu")
            del vae, tokenizer_one,tokenizer_two,tokenizer_three, text_encoder_one,text_encoder_two,text_encoder_three
            gc.collect()
            torch.cuda.empty_cache()
    
    # repeat_datarows = []
    # for datarow in datarows:
    #     for i in range(args.repeats):
    #         repeat_datarows.append(datarow)
    # datarows = repeat_datarows
    
    datarows = datarows * args.repeats
    # resume from cpu after cache files
    transformer.to(accelerator.device)

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

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
        if args.mixed_precision == "bf16":
            try:
                from adamw_bf16 import AdamWBF16
            except ImportError:
                raise ImportError(
                    "To use bf Adam, please install the AdamWBF16 library: `pip install adamw-bf16`."
                )
            optimizer_class = AdamWBF16
            transformer.to(dtype=torch.bfloat16)
        elif use_8bit_adam:
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
        latents = torch.stack([example["latent"] for example in examples])
        # time_ids = torch.stack([example["time_id"] for example in examples])
        prompt_embeds = torch.stack([example["prompt_embed"] for example in examples])
        pooled_prompt_embeds = torch.stack([example["pooled_prompt_embed"] for example in examples])

        return {
            "latents": latents,
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            # "time_ids": time_ids,
        }
    # create dataset based on input_dir
    train_dataset = CachedImageDataset(datarows,conditional_dropout_percent=args.caption_dropout)

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
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )


    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "sd35-lora"
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


    # def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
    #     sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
    #     schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
    #     timesteps = timesteps.to(accelerator.device)

    #     step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    #     sigma = sigmas[step_indices].flatten()
    #     while len(sigma.shape) < n_dim:
    #         sigma = sigma.unsqueeze(-1)
    #     return sigma


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
    
    max_time_steps = noise_scheduler.config.num_train_timesteps
    if args.max_time_steps is not None and args.max_time_steps > 0:
        max_time_steps = args.max_time_steps
        
    
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
                with accelerator.autocast():
                    latents = batch["latents"].to(accelerator.device)
                    prompt_embeds = batch["prompt_embeds"].to(accelerator.device)
                    pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(accelerator.device)
                    
                    
                    noise = torch.randn_like(latents)
                    bsz, _, _, _ = latents.shape
                    
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
                    
                    # Predict the noise residual
                    model_pred = transformer(
                        hidden_states=noisy_model_input,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        return_dict=False,
                    )[0]

                    # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                    # Preconditioning of the model outputs.
                    if args.precondition_outputs:
                        model_pred = model_pred * (-sigmas) + (1.0 - sigmas) * latents + sigmas * noise

                    
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
                    if args.precondition_outputs:
                        target = latents
                    else:
                        target = noise - latents
                    
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
                    logs = {"step_loss": step_loss, lr_name: lr, "epoch": epoch}
                    accelerator.log(logs, step=global_step)
                    progress_bar.set_postfix(**logs)
                    
                    if global_step >= max_train_steps:
                        break
                    del step_loss
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
        
        if accelerator.is_main_process:
            if (epoch >= args.skip_epoch and epoch % args.save_model_epochs == 0) or epoch == args.num_train_epochs - 1:
                # accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"{args.save_name}-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
            
            # only execute when val_metadata_path exists
            if ((epoch >= args.skip_epoch and epoch % args.validation_epochs == 0) or epoch == args.num_train_epochs - 1) and os.path.exists(val_metadata_path):
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
                        validation_dataset = CachedImageDataset(validation_datarows,conditional_dropout_percent=0)
                        
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
                                latents = batch["latents"].to(accelerator.device)
                                noise = torch.randn_like(latents)
                                bsz, _, _, _ = latents.shape
                                
                                prompt_embeds = batch["prompt_embeds"].to(accelerator.device)
                                pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(accelerator.device)
                                
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
                                
                                # Predict the noise residual
                                model_pred = transformer(
                                    hidden_states=noisy_model_input,
                                    timestep=timesteps,
                                    encoder_hidden_states=prompt_embeds,
                                    pooled_projections=pooled_prompt_embeds,
                                    return_dict=False,
                                )[0]

                                # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                                # Preconditioning of the model outputs.
                                if args.precondition_outputs:
                                    model_pred = model_pred * (-sigmas) + noisy_model_input

                                
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
                                if args.precondition_outputs:
                                    target = latents
                                else:
                                    target = noise - latents
                                
                                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                                
                                # Compute regular loss.
                                loss = torch.mean(
                                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                                    1,
                                )
                                loss = loss.mean()

                                
                                total_loss+=loss.detach()
                                del latents, target, loss, model_pred,  timesteps,  bsz, noise, noisy_model_input
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