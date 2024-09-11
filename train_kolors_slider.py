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

# this code inspired from https://github.com/rohitgandikota/sliders codebase
# make a few changes
# changed the training target to kolors
# changed the uncondition embedding from uncondition prompt to opposite prompt
# use accelerator, peft library
# from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.models.model_loading_utils import load_model_dict_into_meta
# import jsonlines

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
    DDPMScheduler,
    # EulerDiscreteScheduler,
    # DiffusionPipeline,
    UNet2DConditionModel,
    SchedulerMixin
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
    is_wandb_available,
)
from diffusers.loaders import LoraLoaderMixin
# from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

# from diffusers import StableDiffusionXLPipeline
from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256 import StableDiffusionXLPipeline
from tqdm import tqdm 
# from PIL import Image 

from sklearn.model_selection import train_test_split

import json


# import sys
from utils.image_utils_kolors import BucketBatchSampler, CachedPairsDataset

# from prodigyopt import Prodigy


# https://github.com/Lightning-AI/pytorch-lightning/blob/0d52f4577310b5a1624bed4d23d49e37fb05af9e/src/lightning_fabric/utilities/seed.py
from random import getstate as python_get_rng_state
from random import setstate as python_set_rng_state

from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from kolors.models.modeling_chatglm import ChatGLMModel
from kolors.models.tokenization_chatglm import ChatGLMTokenizer
# try:
#     from diffusers.utils import randn_tensor
# except:
#     from diffusers.utils.torch_utils import randn_tensor

if is_wandb_available():
    import wandb
    
from safetensors.torch import save_file

from utils.dist_utils import flush

import glob

from diffusers.utils.torch_utils import randn_tensor

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.30.0.dev0")

logger = get_logger(__name__)


from utils.image_utils_kolors import compute_text_embeddings
from utils.dist_utils import flush

from utils.utils import get_md5_by_path
from torchvision import transforms

import cv2

from utils.image_utils_kolors import crop_image
# from slider.lora import LoRANetwork

# import slider.debug_util as debug_util

# =========Debias implementation from: https://github.com/kohya-ss/sd-scripts/blob/main/library/custom_train_functions.py#L99
def prepare_scheduler_for_custom_training(noise_scheduler, device):
    if hasattr(noise_scheduler, "all_snr"):
        return

    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    alpha = sqrt_alphas_cumprod
    sigma = sqrt_one_minus_alphas_cumprod
    all_snr = (alpha / sigma) ** 2

    noise_scheduler.all_snr = all_snr.to(device)

def apply_debiased_estimation(loss, timesteps, noise_scheduler):
    snr_t = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])  # batch_size
    snr_t = torch.minimum(snr_t, torch.ones_like(snr_t) * 1000)  # if timestep is 0, snr_t is inf, so limit it to 1000
    weight = 1 / torch.sqrt(snr_t)
    loss = weight * loss
    return loss
# =========Debias implementation from: https://github.com/kohya-ss/sd-scripts/blob/main/library/custom_train_functions.py#L99

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
        default=50,
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
        default=1,
        help=("skip val and save model before x epochs"),
    )
    parser.add_argument(
        "--skip_step",
        type=int,
        default=1,
        help=("skip val and save model before x step"),
    )
    
    parser.add_argument(
        "--break_epoch",
        type=int,
        default=1,
        help=("break training after x epochs"),
    )
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
    # parser.add_argument(
    #     "--resolution_config",
    #     type=str,
    #     default=None,
    #     help=("default: '1024', accept str: '1024', '2048'"),
    # )
    # parser.add_argument(
    #     "--use_debias",
    #     action="store_true",
    #     help="Use debiased estimation loss",
    # )
    
    # parser.add_argument(
    #     "--snr_gamma",
    #     type=float,
    #     default=5,
    #     help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
    #     "More details here: https://arxiv.org/abs/2303.09556.",
    # )
    
    # parser.add_argument(
    #     "--uncondition_prompt",
    #     type=str,
    #     default="abstruct",
    #     help=(
    #         "the main uncondition prompt for both positive images and negative images"
    #     ),
    # )
    parser.add_argument(
        "--main_prompt",
        type=str,
        default="a girl",
        help=(
            "the main prompt for both positive images and negative images"
        ),
    )
    parser.add_argument(
        "--pos_prompt",
        type=str,
        default="beatiful",
        help=(
            "positive images generation prompt to describe the main subject"
        ),
    )
    parser.add_argument(
        "--neg_prompt",
        type=str,
        default="ugly",
        help=(
            "negative images generation prompt to describe the main subject"
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help=(
            "Image generation steps"
        ),
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=3.5,
        help=(
            "Image generation guidance_scale"
        ),
    )
    parser.add_argument(
        "--generation_batch",
        type=int,
        default=5,
        help=(
            "Image generation batch"
        ),
    )
    
    parser.add_argument(
        "--image_prefix",
        type=str,
        default="image",
        help=(
            "image filename prefix"
        ),
    )
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def concat_embeddings(
    unconditional: torch.FloatTensor,
    conditional: torch.FloatTensor,
    n_imgs: int,
):
    return torch.cat([unconditional, conditional]).repeat_interleave(n_imgs, dim=0)

def predict_noise_xl(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    timestep: int,  # 現在のタイムステップ
    latents: torch.FloatTensor,
    text_embeddings: torch.FloatTensor,  # uncond な text embed と cond な text embed を結合したもの
    add_text_embeddings: torch.FloatTensor,  # pooled なやつ
    add_time_ids: torch.FloatTensor,
    guidance_scale=7.5,
    # guidance_rescale=0.7,
) -> torch.FloatTensor:
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep).to(dtype=text_embeddings.dtype)

    added_cond_kwargs = {
        "text_embeds": add_text_embeddings,
        "time_ids": add_time_ids,
    }

    # predict the noise residual
    noise_pred = unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=text_embeddings,
        added_cond_kwargs=added_cond_kwargs,
    ).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    guided_target = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )

    return guided_target

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
    # revision = None
    # variant = None
    prodigy_decouple = True
    prodigy_beta3 = None
    prodigy_use_bias_correction = True
    prodigy_safeguard_warmup = True
    prodigy_d_coef = 2
    
    
    lr_num_cycles = 1
    lr_power = 1
    
    # this is for consistence validation. all validation would use this seed to generate the same validation set
    # val_seed = random.randint(1, 100)
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
    # args.validation_epochs = 1
    # args.train_batch_size = 1
    # args.repeats = 1
    # args.gradient_accumulation_steps = 1
    # args.use_dora = False
    # args.caption_dropout = 0.2
    
    # args.vae_path = "F:/models/VAE/sdxl_vae.safetensors"
    # args.pretrained_model_name_or_path = "F:/T2ITrainer/kolors_models"
    # # args.train_data_dir = "F:/ImageSet/kolors_slider"
    # args.train_data_dir = "F:/ImageSet/kolors_slider_anime"
    # not use uncondition prompt
    # args.uncondition_prompt = "photo, realistic"
    # args.main_prompt = "anime artwork of a beautiful girl, "
    # args.pos_prompt = "highly detailed, well drawing, digital artwork, detailed background"
    # args.neg_prompt = "sketch, unfinised drawing, monochrome, simple background"
    # args.steps = 30
    # args.cfg = 3.5
    # args.seed = 1
    # args.generation_batch = 5
    
    # args.mixed_precision = "fp16"
    # args.train_batch_size = 1
    # args.output_dir = "F:/models/kolors"
    # args.save_name = "kolors-anime-slider"
    # args.num_train_epochs = 5
    # args.repeats = 100
    # args.recreate_cache = True
    # args.save_model_epochs = 1
    
    default_positive_scale = 2
    default_negative_scale = -2
    
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
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    
    # prepare noise scheduler
    # noise_scheduler = DDPMScheduler(
    #     beta_start=0.00085, beta_end=0.014, beta_schedule="scaled_linear", num_train_timesteps=1100, clip_sample=False, 
    #     dynamic_thresholding_ratio=0.995, prediction_type="epsilon", steps_offset=1, timestep_spacing="leading", trained_betas=None
    # )
    # if args.use_debias:
    #     prepare_scheduler_for_custom_training(noise_scheduler, accelerator.device)
    
    # noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    
    device = accelerator.device
    offload_device = accelerator.device
    
    # create metadata.jsonl if not exist
    metadata_path = os.path.join(args.train_data_dir, f'metadata_kolors_slider.json')
    if not os.path.exists(metadata_path):
        offload_device = torch.device("cpu")
    
    # load from repo
    if args.pretrained_model_name_or_path == "Kwai-Kolors/Kolors":
        unet = UNet2DConditionModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="unet", variant="fp16"
            ).to(offload_device, dtype=weight_dtype)
    else:
        # load from repo
        unet_folder = os.path.join(args.pretrained_model_name_or_path, "unet")
        weight_file = "diffusion_pytorch_model"
        unet_variant = None
        ext = ".safetensors"
        # diffusion_pytorch_model.fp16.safetensors
        fp16_weight = os.path.join(unet_folder, f"{weight_file}.fp16{ext}")
        fp32_weight = os.path.join(unet_folder, f"{weight_file}{ext}")
        if os.path.exists(fp16_weight):
            unet_variant = "fp16"
        elif os.path.exists(fp32_weight):
            unet_variant = None
        else:
            raise FileExistsError(f"{fp16_weight} and {fp32_weight} not found. \n Please download the model from https://huggingface.co/Kwai-Kolors/Kolors or https://hf-mirror.com/Kwai-Kolors/Kolors")
            
        unet = UNet2DConditionModel.from_pretrained(
                    unet_folder, variant=unet_variant
                ).to(offload_device, dtype=weight_dtype)
    
    if not (args.model_path is None or args.model_path == ""):
        # load from file
        state_dict = safetensors.torch.load_file(args.model_path, device="cpu")
        unexpected_keys = load_model_dict_into_meta(
            unet,
            state_dict,
            device=offload_device,
            dtype=torch.float32,
            model_name_or_path=args.model_path,
        )
        # updated_state_dict = unet.state_dict()
        if len(unexpected_keys) > 0:
            print(f"Unexpected keys in state_dict: {unexpected_keys}")
        unet.to(offload_device, dtype=weight_dtype)
        del state_dict,unexpected_keys
        flush()

    unet.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # now we will add new LoRA weights to the attention layers
    # unet_lora_config = LoraConfig(
    #     use_dora=args.use_dora,
    #     r=args.rank,
    #     lora_alpha=args.rank,
    #     init_lora_weights="gaussian",
    #     target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    # )
    # unet.add_adapter(unet_lora_config)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder atten layers
            unet_lora_layers_to_save = None
            for model in models:
                if isinstance(model, type(unwrap_model(unet))):
                    unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            # save all
            StableDiffusionXLPipeline.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save
            )
            
            # save to kohya
            peft_state_dict = convert_all_state_dict_to_peft(unet_lora_layers_to_save)
            kohya_state_dict = convert_state_dict_to_kohya(peft_state_dict)
            # add prefix to keys
            prefix = 'lora_unet_'
            prefixed_state_dict = {prefix + key: value for key, value in kohya_state_dict.items()}
            last_part = os.path.basename(os.path.normpath(output_dir))
            file_path = f"{output_dir}/{last_part}.safetensors"
            # save comfyui/webui lora as the name of parent
            save_file(prefixed_state_dict, file_path)

    def load_model_hook(models, input_dir):
        unet_ = None
        while len(models) > 0:
            model = models.pop()
            if isinstance(model, type(unwrap_model(unet))):
                unet_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)

        unet_state_dict = {f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")}
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict, adapter_name="default")
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
            models = [unet_]
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)


    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # now we will add new LoRA weights to the attention layers
    unet_lora_config = LoraConfig(
        use_dora=args.use_dora,
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)
    
    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [unet]
        # # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)
        # network.cast_training_params()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True


    unet_lora_parameters = list(filter(lambda p: p.requires_grad, unet.parameters()))
    # Optimization parameters
    unet_lora_parameters_with_lr = {"params": unet_lora_parameters, "lr": args.learning_rate}
    params_to_optimize = [unet_lora_parameters_with_lr]
    
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
            unet.to(dtype=torch.bfloat16)
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
    
    # ==========================================================
    # Create train dataset
    # ==========================================================
    
    datarows = []
    
    with torch.no_grad():
        if args.train_data_dir is not None:
            if not os.path.exists(args.train_data_dir):
                raise FileNotFoundError(f"{args.train_data_dir} does not exist")
            input_dir = args.train_data_dir
            recreate_cache = args.recreate_cache
            
            # ensure both dir exist in training dir
            pos_dir = os.path.join(input_dir, "positive")
            if not os.path.exists(pos_dir):
                raise FileNotFoundError(f"{pos_dir} does not exist")
            
            neg_dir = os.path.join(input_dir, "negative")
            if not os.path.exists(neg_dir):
                raise FileNotFoundError(f"{neg_dir} does not exist")
            
            supported_image_types = ['.jpg','.jpeg','.png','.webp']
            pos_files = glob.glob(f"{pos_dir}/**", recursive=True)
            pos_image_files = [f for f in pos_files if os.path.splitext(f)[-1].lower() in supported_image_types]
            neg_files = glob.glob(f"{neg_dir}/**", recursive=True)
            neg_image_files = [f for f in neg_files if os.path.splitext(f)[-1].lower() in supported_image_types]
            
            file_list = [pos_image_files, neg_image_files]
            pos_len = len(pos_image_files)
            neg_len = len(neg_image_files)
            
            if pos_len == 0 or neg_len == 0:
                raise ValueError("No images found in the specified directories.")
            
            if pos_len != neg_len:
                print("Number of positive and negative images do not match. Using the minimum number of images.")
            
            
            metadata = {}
            # single_image_training = False
            if os.path.exists(metadata_path) and not recreate_cache:
                with open(metadata_path, "r", encoding='utf-8') as readfile:
                    metadata = json.loads(readfile.read())
                    # check md5
                    # function to remove metadata datarows which in not exist in directory
                    def align_metadata(datarows,image_files):
                        new_metadatarows = []
                        for metadata_datarow in datarows:
                            # if some row not in current image_files, ignore it
                            if metadata_datarow['image_path'] in image_files:
                                new_metadatarows.append(metadata_datarow)
                        return new_metadatarows

                    # filter out metadata rows that are not in current image_files
                    for i,image_files in enumerate(file_list):
                        generation_config = metadata['generation_configs'][i]
                        if 'item_list' in generation_config:
                            metadata['generation_configs'][i]['item_list'] = align_metadata(generation_config['item_list'],image_files)
                    
                    
            else:
                metadata = {
                    'main_prompt': args.main_prompt,
                    # 'uncondition_prompt': args.uncondition_prompt,
                    'pos_prompt': args.pos_prompt,
                    'neg_prompt': args.neg_prompt,
                    'generation_batch': args.generation_batch,
                    'pretrained_model_name_or_path': args.pretrained_model_name_or_path,
                    'steps': args.steps,
                    'cfg': args.cfg,
                    'seed': args.seed,
                    "generation_configs":[
                        {
                            "set_name":"positive",
                            "prompt":f"{args.main_prompt}, {args.pos_prompt}",
                        },
                        {
                            "set_name":"negative",
                            "prompt":f"{args.main_prompt}, {args.neg_prompt}",
                        },
                        {
                            "set_name":"main",
                            "prompt":f"{args.main_prompt}",
                        },
                    ],
                }
            
            text_encoder = None
            tokenizer = None
            vae = None
            
            prompt_embeds_list = []
            for generation_config in metadata['generation_configs']:
                set_name = generation_config['set_name']
                npz_path = f"{args.train_data_dir}/{set_name}.npkolors"
                # recreate_cache = False
                if os.path.exists(npz_path):
                    if 'npz_path_md5' in generation_config:
                        if generation_config['npz_path_md5'] != get_md5_by_path(npz_path):
                            recreate_cache = True
                            print("npz_path_md5 changed, recreating cache")
                    if not recreate_cache:
                        npz_path_md5 = get_md5_by_path(npz_path)
                        generation_config['npz_path'] = npz_path
                        generation_config['npz_path_md5'] = npz_path_md5
                        cached_npz = torch.load(npz_path)
                        prompt_embeds = cached_npz['prompt_embed']
                        pooled_prompt_embeds = cached_npz['pooled_prompt_embed']
                        prompt_embeds_list.append((set_name,prompt_embeds, pooled_prompt_embeds))
                        continue
                
                if text_encoder is None:
                    text_encoder = ChatGLMModel.from_pretrained(
                    f'{args.pretrained_model_name_or_path}/text_encoder',
                    torch_dtype=torch.float16).half().to(device)
                    text_encoder.requires_grad_(False)
                    text_encoder.to(accelerator.device, dtype=weight_dtype)
                
                if tokenizer is None:
                    tokenizer = ChatGLMTokenizer.from_pretrained(f'{args.pretrained_model_name_or_path}/text_encoder')
                
                if vae is None:
                    vae_folder = os.path.join(args.pretrained_model_name_or_path, "vae")
                    if args.vae_path:
                        vae = AutoencoderKL.from_single_file(
                            args.vae_path,
                            config=vae_folder,
                        )
                    else:
                        # load from repo
                        weight_file = "diffusion_pytorch_model"
                        vae_variant = None
                        ext = ".safetensors"
                        # diffusion_pytorch_model.fp16.safetensors
                        fp16_weight = os.path.join(vae_folder, f"{weight_file}.fp16{ext}")
                        fp32_weight = os.path.join(vae_folder, f"{weight_file}{ext}")
                        if os.path.exists(fp16_weight):
                            vae_variant = "fp16"
                        elif os.path.exists(fp32_weight):
                            vae_variant = None
                        else:
                            raise FileExistsError(f"{fp16_weight} and {fp32_weight} not found. \n Please download the model from https://huggingface.co/Kwai-Kolors/Kolors or https://hf-mirror.com/Kwai-Kolors/Kolors")
                            
                        vae = AutoencoderKL.from_pretrained(
                                args.pretrained_model_name_or_path, variant=vae_variant
                            )
                
                    vae.to(accelerator.device, dtype=weight_dtype)
                    vae.requires_grad_(False)
                
                prompt = generation_config['prompt']
                set_name = generation_config['set_name']
                # for positive images generation
                prompt_embeds, pooled_prompt_embeds = compute_text_embeddings([text_encoder],[tokenizer],prompt,device=text_encoder.device)
                prompt_embed = prompt_embeds.squeeze(0)
                pooled_prompt_embed = pooled_prompt_embeds.squeeze(0)
                # save embeddings
                npz_dict = {
                    "prompt_embed": prompt_embed.cpu(), 
                    "pooled_prompt_embed": pooled_prompt_embed.cpu(),
                }
                # save latent to cache file
                torch.save(npz_dict, npz_path)
                generation_config['npz_path'] = npz_path
                npz_path_md5 = get_md5_by_path(npz_path)
                generation_config['npz_path_md5'] = npz_path_md5
                prompt_embeds_list.append((set_name,prompt_embeds, pooled_prompt_embeds))
            
            # not use uncondition
            _, main_prompt_embeds, main_pooled_prompt_embeds = prompt_embeds_list.pop()
            
            del text_encoder, tokenizer
            flush()
            for i in range(len(prompt_embeds_list)):
                metadata['generation_configs'][i]['item_list'] = []
                set_name, prompt_embeds, pooled_prompt_embeds = prompt_embeds_list[i]
                image_files = file_list[i]
                save_dir = f"{args.train_data_dir}/{set_name}"
                for image_path in image_files:
                    filename, ext = os.path.splitext(os.path.basename(image_path))
                    latent_path = f"{save_dir}/{filename}.nplatent"
                    
                    try:
                        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                        if image is not None:
                            # Convert to RGB format (assuming the original image is in BGR)
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        else:
                            print(f"Failed to open {image_path}.")
                    except Exception as e:
                        print(f"An error occurred while processing {image_path}: {e}")
                        
                    height, width, _ = image.shape
                    # recreate_cache = False
                    # check md5 on exists latent
                    if os.path.exists(latent_path):
                        generation_config = metadata['generation_configs'][i]
                        if 'latent_path_md5' in generation_config:
                            if generation_config['latent_path_md5'] != get_md5_by_path(latent_path):
                                recreate_cache = True
                        if not recreate_cache:
                            # metadata['generation_configs'][i]['latent_path_md5'] = get_md5_by_path(latent_path)
                            training_item = {
                                'bucket':f"{width}x{height}",
                                'latent_path':latent_path,
                                'latent_path_md5':get_md5_by_path(latent_path),
                                'image_path':image_path,
                                'image_path_md5':get_md5_by_path(image_path),
                            }
                            metadata['generation_configs'][i]['item_list'].append(training_item)
                            continue
                    
                    # save latent
                    latent_path = f"{save_dir}/{filename}.nplatent"
                
                    original_size = (height, width)
                    
                    cropped_image,crop_x,crop_y = crop_image(image)
                    crops_coords_top_left = (crop_y,crop_x)
                    
                    
                    image_height, image_width, _ = cropped_image.shape
                    target_size = (image_height,image_width)
                    
                    # vae encode file
                    train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
                    image = train_transforms(cropped_image)
                    
                    # create tensor latent
                    pixel_values = []
                    pixel_values.append(image)
                    pixel_values = torch.stack(pixel_values).to(vae.device, dtype=vae.dtype)
                    with torch.no_grad():
                        #contiguous_format = (contiguous memory block), unsqueeze(0) adds bsz 1 dimension, else error: but got weight of shape [128] and input of shape [128, 512, 512]
                        latent = vae.encode(pixel_values).latent_dist.sample().squeeze(0)
                        latent = latent * vae.config.scaling_factor
                        del pixel_values
                        
                    # check latent
                    # latent = latent.unsqueeze(0).to(torch.float16)
                    # image_processor = VaeImageProcessor(vae_scale_factor=vae.config.scaling_factor)
                    # with torch.no_grad():
                    #     image = vae.decode(latent / vae.config.scaling_factor, return_dict=False)[0]
                    # image = image_processor.postprocess(image, output_type="pil")[0]
                    # image.save("test.png")
                    
                    time_id = torch.tensor(list(original_size + crops_coords_top_left + target_size))

                    latent_dict = {
                        'time_id': time_id.cpu(),
                        'latent': latent.cpu()
                    }
                    torch.save(latent_dict, latent_path)
                
                    
                    training_item = {
                        'bucket':f"{image_width}x{image_height}",
                        'latent_path':latent_path,
                        'latent_path_md5':get_md5_by_path(latent_path),
                        'image_path':image_path,
                        'image_path_md5':get_md5_by_path(image_path),
                    }
                    metadata['generation_configs'][i]['item_list'].append(training_item)
                    
            del vae
            flush()
            # save metadata
            with open(metadata_path, "w", encoding='utf-8') as writefile:
                writefile.write(json.dumps(metadata, indent=4))
                
            # prepare datarows, map pos and neg in onw single row
            generation_configs = metadata['generation_configs']
            pos_config = generation_configs[0]
            neg_config = generation_configs[1]
            main_config = generation_configs[2]
            if len(pos_config['item_list']) == 0 and len(neg_config['item_list']) == 0:
                raise ValueError("No item in metadata.")
            if len(pos_config['item_list']) != len(neg_config['item_list']):
                raise ValueError("Positive and Negative images must have same number of images")
            for pos_item, neg_item in zip(pos_config['item_list'], neg_config['item_list']):
                datarows.append(
                    {
                        "bucket": pos_item['bucket'],
                        "main_npz_path": main_config['npz_path'],
                        "pos_npz_path": pos_config['npz_path'],
                        "pos_latent_path": pos_item['latent_path'],
                        "neg_npz_path": neg_config['npz_path'],
                        "neg_latent_path": neg_item['latent_path'],
                    }
                )
            
            # no validation in this training
            # training_datarows, validation_datarows = train_test_split(datarows, train_size=1, test_size=0)
            # datarows = training_datarows
        
        
    # repeat_datarows = []
    # for datarow in datarows:
    #     for i in range(args.repeats):
    #         repeat_datarows.append(datarow)
    # datarows = repeat_datarows
    datarows = datarows * args.repeats
    # resume from cpu after cache files
    unet.to(accelerator.device)


    def collate_fn(examples):
        pos_latents = torch.stack([example["pos_latent"] for example in examples])
        pos_prompt_embeds = torch.stack([example["pos_prompt_embed"] for example in examples])
        pos_pooled_prompt_embeds = torch.stack([example["pos_pooled_prompt_embed"] for example in examples])
        pos_time_ids = torch.stack([example["pos_time_id"] for example in examples])
        neg_latents = torch.stack([example["neg_latent"] for example in examples])
        neg_prompt_embeds = torch.stack([example["neg_prompt_embed"] for example in examples])
        neg_pooled_prompt_embeds = torch.stack([example["neg_pooled_prompt_embed"] for example in examples])
        neg_time_ids = torch.stack([example["neg_time_id"] for example in examples])
        main_prompt_embeds = torch.stack([example["main_prompt_embed"] for example in examples])
        main_pooled_prompt_embeds = torch.stack([example["main_pooled_prompt_embed"] for example in examples])

        return {
            "pos_latents":pos_latents,
            "pos_prompt_embeds":pos_prompt_embeds,
            "pos_pooled_prompt_embeds":pos_pooled_prompt_embeds,
            "pos_time_ids":pos_time_ids,
            "neg_latents":neg_latents,
            "neg_prompt_embeds":neg_prompt_embeds,
            "neg_pooled_prompt_embeds":neg_pooled_prompt_embeds,
            "neg_time_ids":neg_time_ids,
            "main_prompt_embeds":main_prompt_embeds,
            "main_pooled_prompt_embeds":main_pooled_prompt_embeds,
        }
        
    # create dataset based on input_dir
    # no caption dropout for slider training, avoid affect the uncondition space
    train_dataset = CachedPairsDataset(datarows,conditional_dropout_percent=0)

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
    
    # ================================================================
    # End Create train dataset
    # ================================================================
    
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
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "kolors-slider-lora"
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
    
    device = accelerator.device
    max_denoising_steps = 50
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        # ================================================
        # loop over dataloader
        # ================================================
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                with accelerator.autocast():
                    pos_latents = batch["pos_latents"].to(accelerator.device)
                    pos_prompt_embeds = batch["pos_prompt_embeds"].to(accelerator.device)
                    pos_pooled_prompt_embeds = batch["pos_pooled_prompt_embeds"].to(accelerator.device)
                    pos_time_ids = batch["pos_time_ids"].to(accelerator.device)
                    neg_latents = batch["neg_latents"].to(accelerator.device)
                    neg_prompt_embeds = batch["neg_prompt_embeds"].to(accelerator.device)
                    neg_pooled_prompt_embeds = batch["neg_pooled_prompt_embeds"].to(accelerator.device)
                    neg_time_ids = batch["neg_time_ids"].to(accelerator.device)
                    main_prompt_embeds = batch["main_prompt_embeds"].to(accelerator.device)
                    main_pooled_prompt_embeds = batch["main_pooled_prompt_embeds"].to(accelerator.device)
                    
                    # prepare predicted noise image
                    # 
                    noise_scheduler.set_timesteps(
                        max_denoising_steps, device=device
                    )
                    optimizer.zero_grad()
                    # 1 ~ 49 からランダム
                    timesteps_to = torch.randint(
                        1, max_denoising_steps, (1,)
                    ).item()
                    
                    shape = pos_latents.shape

                    seed = random.randint(0,2*15)
                    
                    # get positive latents
                    generator = torch.manual_seed(seed)
                    noise = randn_tensor(shape, generator=generator, device=device)
                    timestep = noise_scheduler.timesteps[timesteps_to:timesteps_to+1]
                    # get latents
                    pos_noised_latents = noise_scheduler.add_noise(pos_latents, noise, timestep)
                    
                    pos_noised_latents = pos_noised_latents.to(device, dtype=weight_dtype)
                    noise = noise.to(device, dtype=weight_dtype)
                    
                    # get negative latents
                    generator = torch.manual_seed(seed)
                    # use the same noise and timestep as positive
                    # noise = randn_tensor(shape, generator=generator, device=device)
                    # timestep = noise_scheduler.timesteps[timesteps_to:timesteps_to+1]
                    # get latents
                    neg_noised_latents = noise_scheduler.add_noise(neg_latents, noise, timestep)
                    neg_noised_latents = neg_noised_latents.to(device, dtype=weight_dtype)
                    # noise = noise.to(device, dtype=weight_dtype)
                    
                    # reset noise_scheduler to 1000
                    noise_scheduler.set_timesteps(noise_scheduler.config.num_train_timesteps)
                    current_timestep = noise_scheduler.timesteps[
                        int(timesteps_to * noise_scheduler.config.num_train_timesteps / max_denoising_steps)
                    ]

                    # # scale lora 
                    unet.set_adapters('default', default_positive_scale)
                    pos_target_latents = predict_noise_xl(
                        unet,
                        noise_scheduler,
                        current_timestep,
                        pos_noised_latents,
                        text_embeddings=concat_embeddings(
                            neg_prompt_embeds,
                            pos_prompt_embeds,
                            1,
                        ),
                        add_text_embeddings=concat_embeddings(
                            neg_pooled_prompt_embeds,
                            pos_pooled_prompt_embeds,
                            1,
                        ),
                        add_time_ids=concat_embeddings(
                            pos_time_ids, pos_time_ids, 1
                        ),
                        guidance_scale=1,
                    )

                    loss_pos = F.mse_loss(pos_target_latents.float(), noise.float())
                    # progress_bar.set_description(f"pos_step_loss*1k: {loss_pos.item():.4f}")
                    
                    # Backpropagate
                    accelerator.backward(loss_pos)

                    pos_step_loss = loss_pos.detach().item()
                    lr = lr_scheduler.get_last_lr()[0]
                    lr_name = "lr"
                    if args.optimizer == "prodigy":
                        if resume_step>0 and resume_step == global_step:
                            lr = 0
                        else:
                            lr = lr_scheduler.optimizers[-1].param_groups[0]["d"] * lr_scheduler.optimizers[-1].param_groups[0]["lr"]
                        lr_name = "lr/d*lr"
                    logs = {"pos_step_loss": pos_step_loss, lr_name: lr, "epoch": epoch}
                    accelerator.log(logs, step=global_step)
                    progress_bar.set_postfix(**logs)
                    
                    
                    unet.set_adapters('default', default_negative_scale)
                    # scale lora 
                    neg_target_latents = predict_noise_xl(
                        unet,
                        noise_scheduler,
                        current_timestep,
                        neg_noised_latents,
                        text_embeddings=concat_embeddings(
                            neg_prompt_embeds,
                            main_prompt_embeds,
                            1,
                        ),
                        add_text_embeddings=concat_embeddings(
                            neg_pooled_prompt_embeds,
                            main_pooled_prompt_embeds,
                            1,
                        ),
                        add_time_ids=concat_embeddings(
                            neg_time_ids, neg_time_ids, 1
                        ),
                        guidance_scale=1,
                    )

                    loss_neg = F.mse_loss(neg_target_latents.float(), noise.float())
                    # progress_bar.set_description(f"neg_step_loss*1k: {loss_neg.item()*1000:.4f}")
                    
                    # Backpropagate
                    accelerator.backward(loss_neg)

                    neg_step_loss = loss_neg.detach().item()
                    lr = lr_scheduler.get_last_lr()[0]
                    lr_name = "lr"
                    if args.optimizer == "prodigy":
                        if resume_step>0 and resume_step == global_step:
                            lr = 0
                        else:
                            lr = lr_scheduler.optimizers[-1].param_groups[0]["d"] * lr_scheduler.optimizers[-1].param_groups[0]["lr"]
                        lr_name = "lr/d*lr"
                    logs = {"neg_step_loss": neg_step_loss, lr_name: lr, "epoch": epoch}
                    accelerator.log(logs, step=global_step)
                    progress_bar.set_postfix(**logs)
                    
                    if accelerator.sync_gradients:
                        params_to_clip = unet_lora_parameters
                        accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
                        
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    #post batch check for gradient updates
                    accelerator.wait_for_everyone()
                    if accelerator.sync_gradients:
                        progress_bar.update(1)
                        global_step += 1
                    
                    if global_step >= max_train_steps:
                        break
                    # del step_loss
                    flush()
    
        if global_step < args.skip_step:
            continue
        
        
        # store rng before validation
        before_state = torch.random.get_rng_state()
        np_seed = np.random.seed()
        py_state = python_get_rng_state()
        
        if accelerator.is_main_process:
            if (epoch >= args.skip_epoch and epoch % args.save_model_epochs == 0) or epoch == args.num_train_epochs - 1:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"{args.save_name}-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
                    

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