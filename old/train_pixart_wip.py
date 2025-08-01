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

# before 20240401, initial codebase
# 20240401, added caption ext support
#           need to handle multi batch size and gas for dataset stack images.
#           need to implement validation loss
#           need to add te training
# 20240402 bucketing works!!!!, many thanks to @minienglish1 from everydream discord
#          added whole repeats to dataset
from diffusers.models.attention_processor import AttnProcessor2_0
# import jsonlines

import datetime
import time

import argparse
import functools
import gc
# import logging
import math
import os
import random
import shutil
from pathlib import Path
from mmcv.runner import LogBuffer

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
# from compel import Compel, ReturnedEmbeddingsType

from sklearn.model_selection import train_test_split

from transformers import T5EncoderModel, T5Tokenizer

import json


# import sys
# sys.path.append('F:/T2ITrainer/utils')

from utils.misc import set_random_seed
from utils.builder import build_model

import utils.pixart_image_utils
from utils.pixart_image_utils import CachedImageDataset
from utils.bucket.bucket_batch_sampler import BucketBatchSampler

from utils.iddpm import IDDPM
from utils.dpm_solver import DPMS

from utils.pixart_checkpoint import save_checkpoint, load_checkpoint

from utils.optimizer import build_optimizer
from utils.lr_scheduler import build_lr_scheduler


from utils.PixArtMS import PixArtMS
from torch.utils.data import default_collate

# from meta
# https://github.com/facebookresearch/schedule_free
import schedulefree

from prodigyopt import Prodigy

# https://github.com/Lightning-AI/pytorch-lightning/blob/0d52f4577310b5a1624bed4d23d49e37fb05af9e/src/lightning_fabric/utilities/seed.py
from random import getstate as python_get_rng_state
from random import setstate as python_set_rng_state

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.27.0.dev0")

logger = get_logger(__name__)

DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}



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
    # parser.add_argument(
    #     "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    # )
    # parser.add_argument(
    #     "--caption_column",
    #     type=str,
    #     default="text",
    #     help="The column of the dataset containing a caption or a list of captions.",
    # )
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
    parser.add_argument(
        "--caption_exts",
        type=str,
        default=".txt",
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--validation_ratio",
        type=float,
        default=0.1,
        help=(
            "validation_ratio split ratio of validation set for val/loss"
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
    args.seed = 4321
    # args.output_dir = 'F:/models/unet/output'
    args.output_dir = 'F:/models/dit/output'
    if not os.path.exists(args.output_dir):
        # create dir
        os.makedirs(args.output_dir,exist_ok=True)
    args.logging_dir = 'logs'
    args.mixed_precision = "fp16"
    # args.report_to = "tensorboard"
    
    args.report_to = "wandb"
    args.enable_xformers_memory_efficient_attention = True
    args.gradient_checkpointing = True
    args.allow_tf32 = True


    args.lr_warmup_steps = 1
    # reduce gas from 500 to 100
    args.gradient_accumulation_steps = 1
    # increase save steps from 50 to 250
    args.checkpointing_steps = 90
    args.resume_from_checkpoint = ""
    args.save_name = "openxl_pixart"
    # args.lr_scheduler = "constant"
    args.lr_scheduler = "cosine"
    args.scale_lr = False
    args.use_8bit_adam = True
    args.adam_beta1 = 0.9
    # args.adam_beta2 = 0.999
    args.adam_beta2 = 0.99

    args.adam_weight_decay = 1e-2
    args.adam_epsilon = 1e-08
    args.dataset_name = None
    args.cache_dir = None
    args.caption_column = None
    # args.image_column = 'image'
    # args.caption_column = 'text'
    args.resolution = 1024
    args.center_crop = False
    args.max_train_samples = None
    args.proportion_empty_prompts = 0
    args.dataloader_num_workers = 0
    args.max_train_steps = None

    args.timestep_bias_portion = 0.25
    args.timestep_bias_end = 1000
    args.timestep_bias_begin = 0
    args.timestep_bias_multiplier = 1.0
    args.timestep_bias_strategy = "none"
    args.max_grad_norm = 1.0
    args.validation_prompt = "cosplay photo, A female character in a unique outfit, holding two large, serrated weapons. The character has silver hair, wears a blue dress with black accents, and has a white flower accessory in her hair. The background is minimalistic, featuring a white floor and a few white flowers. The composition is dynamic, with the character positioned in a mid-action pose, and the perspective is from a frontal angle, emphasizing the character's stature and the weapons she wields. 1girl, kaine_(nier), solo, weapon, dual_wielding, underwear, bandages, high_heels, holding, flower, white_hair, gloves, sword, white_panties, breasts, panties, negligee, bandaged_leg, full_body, holding_weapon, hair_ornament, bandaged_arm, lingerie, thigh_strap, hair_flower, holding_sword, "
    
    args.validation_ratio = 0.1
    args.num_validation_images = 1
    args.caption_exts = '.txt,.wd14_cap'
    args.pipeline_load_from = "F:/PixArt-sigma/output/pixart_sigma_sdxlvae_T5_diffusers"


    image_size = args.resolution  # @param [256, 512]
    latent_size = int(image_size) // 8
    pred_sigma = True
    learn_sigma = True and pred_sigma
    max_length = 300
    kv_compress_config = None
    visualize = True
    pe_interpolation = 2.0
    qk_norm = False
    micro_condition = False
    model_name = 'PixArtMS_XL_2'
    grad_checkpointing = True
    fp32_attention = True
    snr_loss = True
    train_sampling_steps = 1000

    # num_epochs = 20
    # save_model_steps = 1000
    log_interval = 20
    gradient_clip = 0.01
    
    
    args.model_path = 'F:/models/Stable-diffusion/pixart/PixArt-Sigma-XL-2-1024-MS.pth'
    # args.model_path = 'F:/models/Stable-diffusion/pixart/epoch_280_step_281.pth'
    
    # args.train_data_dir = 'F:/ImageSet/openxl2_realism'
    # try to use clip filtered dataset
    # args.train_data_dir = 'F:/ImageSet/openxl2_realism_above_average' 
    args.train_data_dir = "F:/ImageSet/pixart_test_cropped"
    # args.train_data_dir = "F:/ImageSet/pixart_test_one"
    # args.train_data_dir = "F:/ImageSet/openxl2_reg_test"
    # args.num_train_epochs = 2
    # save_model_epochs = 1
    # skip_epoch = 0
    args.num_train_epochs = 30
    save_model_epochs = 5
    args.validation_epochs = 1
    skip_epoch = 0
    break_epoch = 0
    
    # snr_ratio = 0.5
    # snr_epoch = math.ceil(args.num_train_epochs * snr_ratio)
    # args.num_train_epochs = 1
    # save_model_epochs = 1
    # skip_epoch = 0
    # reduce lr from 1e-5 to 2e-6
    # args.learning_rate = 1
    # args.train_batch_size = 1
    # args.learning_rate = 1e-4
    # args.train_batch_size = 15
    args.learning_rate = 1e-5
    args.train_batch_size = 10
    
    # optimizer_config = dict(type='CAMEWrapper', lr=1e-5, weight_decay=0.0, betas=(0.9, 0.999, 0.9999), eps=(1e-30, 1e-16))
    optimizer_config = dict(type='Lion', lr=args.learning_rate, weight_decay=0.01, betas=(0.9, 0.99))

    
    # resume_from_path = "F:/models/Stable-diffusion/pixart/epoch_99_step_100.pth"
    resume_from_path = None
    # resume_from_path = "F:/models/Stable-diffusion/pixart/1_image_baked_v2_epoch_260_step_261.pth"
    # skip_step = 100
    skip_step = 0

    vae_path = "F:/models/VAE/sdxl_vae_fp16fix.safetensors"

    

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb
    # Make one log on every process with the configuration for debugging.
    # logging.basicConfig(
    #     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    #     datefmt="%m/%d/%Y %H:%M:%S",
    #     level=logging.INFO,
    # )
    
    logging_dir = Path(args.output_dir, args.logging_dir)
    # create accelerator
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    
    # accelerator = Accelerator(
    #     mixed_precision=config.mixed_precision,
    #     gradient_accumulation_steps=config.gradient_accumulation_steps,
    #     log_with=args.report_to,
    #     project_dir=os.path.join(config.work_dir, "logs"),
    #     fsdp_plugin=fsdp_plugin,
    #     even_batches=even_batches,
    #     kwargs_handlers=[init_handler]
    # )
    
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    
    # set_seed(args.seed)
    set_random_seed(args.seed)
    


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


    # prepare validation embeddings, it should be prepared before training, using image_utils.py
    # if visualize:
        # preparing embeddings for visualization. We put it here for saving GPU memory
        # validation_prompts = [
        #     "dog",
        #     "portrait photo of a girl, photograph, highly detailed face, depth of field",
        #     "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
        #     "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
        #     "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
        # ]
        # skip = True
        # for prompt in validation_prompts:
        #     if not (os.path.exists(f'output/tmp/{prompt}_{max_length}token.pth')
        #             and os.path.exists(f'output/pretrained_models/null_embed_diffusers_{max_length}token.pth')):
        #         skip = False
        #         logger.info("Preparing Visualization prompt embeddings...")
        #         break
        # if accelerator.is_main_process and not skip:
        #     if config.data.load_t5_feat and (tokenizer is None or text_encoder is None):
        #         logger.info(f"Loading text encoder and tokenizer from {args.pipeline_load_from} ...")
        #         tokenizer = T5Tokenizer.from_pretrained(args.pipeline_load_from, subfolder="tokenizer")
        #         text_encoder = T5EncoderModel.from_pretrained(
        #             args.pipeline_load_from, subfolder="text_encoder", torch_dtype=torch.float16).to(accelerator.device)
        #     for prompt in validation_prompts:
        #         txt_tokens = tokenizer(
        #             prompt, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
        #         ).to(accelerator.device)
        #         caption_emb = text_encoder(txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0]
        #         torch.save(
        #             {'caption_embeds': caption_emb, 'emb_mask': txt_tokens.attention_mask},
        #             f'output/tmp/{prompt}_{max_length}token.pth')
        #     null_tokens = tokenizer(
        #         "", max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
        #     ).to(accelerator.device)
        #     null_token_emb = text_encoder(null_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0]
        #     torch.save(
        #         {'uncond_prompt_embeds': null_token_emb, 'uncond_prompt_embeds_mask': null_tokens.attention_mask},
        #         f'output/pretrained_models/null_embed_diffusers_{max_length}token.pth')
        #     if config.data.load_t5_feat:
        #         del tokenizer
        #         del txt_tokens
        #     flush()


    model_kwargs={"pe_interpolation": pe_interpolation, 
                #   "config":config,
                  "model_max_length": max_length, "qk_norm": qk_norm,
                  "kv_compress_config": kv_compress_config, "micro_condition": micro_condition}

    # build models
    train_diffusion = IDDPM(str(train_sampling_steps), noise_schedule="linear", learn_sigma=learn_sigma, pred_sigma=pred_sigma, snr=snr_loss)
    
    # reference from kohya ss custom_train_functions.py
    # def prepare_scheduler_for_custom_training(noise_scheduler, device):
    #     if hasattr(noise_scheduler, "all_snr"):
    #         return

    #     alphas_cumprod = torch.from_numpy(noise_scheduler.alphas_cumprod)
    #     sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    #     sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    #     alpha = sqrt_alphas_cumprod
    #     sigma = sqrt_one_minus_alphas_cumprod
    #     all_snr = (alpha / sigma) ** 2

    #     noise_scheduler.all_snr = all_snr.to(device)
    
    # prepare_scheduler_for_custom_training(train_diffusion,accelerator.device)
    
    model = build_model(model_name,
                        grad_checkpointing,
                        fp32_attention,
                        input_size=latent_size,
                        learn_sigma=learn_sigma,
                        pred_sigma=pred_sigma,
                        **model_kwargs).train()
    pe_interpolation = {256: 0.5, 512: 1, 1024: 2, 2048: 4}
    # model = PixArtMS(depth=28, hidden_size=1152, patch_size=2, num_heads=16, 
    #         input_size=latent_size,
    #         learn_sigma=learn_sigma,
    #         pred_sigma=pred_sigma,
    #         **model_kwargs
    #     ).train()
    model.to(dtype=weight_dtype)
    logger.info(f"{model.__class__.__name__} Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    missing, unexpected = load_checkpoint(
            args.model_path, model, load_ema=False, max_length=max_length)
    logger.warning(f'Missing keys: {missing}')
    logger.warning(f'Unexpected keys: {unexpected}')

    # build optimizer and lr scheduler
    # lr_scale_ratio = 1
    # if config.get('auto_lr', None):
    #     lr_scale_ratio = auto_scale_lr(config.train_batch_size * get_world_size() * config.gradient_accumulation_steps,
    #                                    config.optimizer, **config.auto_lr)
    optimizer = build_optimizer(model, optimizer_config)
    # weight_decay = 0.01
    # optimizer = Prodigy(model.parameters(), 
    #                     lr=args.learning_rate, 
    #                     safeguard_warmup=True,
    #                     use_bias_correction=True,
    #                     weight_decay=weight_decay,
    #                     betas=(0.9, 0.99),
    #                     decouple=True,
    #                     d_coef=2
    #                     )
    start_epoch = 0
    start_step = 0

    #     logger.warning(f'Missing keys: {missing}')
    #     logger.warning(f'Unexpected keys: {unexpected}')


    # pipeline = StableDiffusionXLPipeline.from_single_file(
    # args.model_path,variant=weight_dtype, use_safetensors=True, 
    # torch_dtype=weight_dtype).to(accelerator.device)
    # # print('pipeline:')
    # # print(pipeline)

    # text_encoder_one = pipeline.text_encoder
    # text_encoder_two = pipeline.text_encoder_2
    # vae = pipeline.vae
    
    # # vae = AutoencoderKL.from_single_file(
    # #     vae_path
    # # )
    # unet = pipeline.unet
    # # print(type(unet))


    if accelerator.mixed_precision == "fp16":
        # from kohya_ss train_util
        org_unscale_grads = accelerator.scaler._unscale_grads_
        def _unscale_grads_replacer(optimizer, inv_scale, found_inf, allow_fp16):
            return org_unscale_grads(optimizer, inv_scale, found_inf, True)

        accelerator.scaler._unscale_grads_ = _unscale_grads_replacer

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True


    # ==========================================================
    # Create train dataset
    # ==========================================================
    # data_files = {}
    # this part need more work afterward, you need to prepare 
    # the train files and val files split first before the training
    # create metadata.jsonl if not exist
    
    rng_state = torch.get_rng_state()
    metadata_path = os.path.join(args.train_data_dir, 'metadata.json')
    val_metadata_path =  os.path.join(args.train_data_dir, 'val_metadata.json')
    if args.train_data_dir is not None:
        # data_files["train"] = os.path.join(args.train_data_dir, "**")
        datarows = []

        if not os.path.exists(metadata_path):
            vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=torch.float16).to(accelerator.device)
            tokenizer = text_encoder = None
            tokenizer = T5Tokenizer.from_pretrained(args.pipeline_load_from, subfolder="tokenizer")
            text_encoder = T5EncoderModel.from_pretrained(
                args.pipeline_load_from, subfolder="text_encoder", torch_dtype=torch.float16).to(accelerator.device)

            # create metadata and latent cache
            datarows = utils.pixart_image_utils.create_metadata_cache(tokenizer,text_encoder,vae,args.train_data_dir)
            
            
            validation_datarows = []
            # prepare validation_slipt
            if args.validation_ratio > 0:
                # buckets = image_utils.get_buckets()
                train_ratio = 1 - args.validation_ratio
                validation_ratio = args.validation_ratio
                if len(datarows) == 1:
                    datarows = datarows + datarows.copy()
                    validation_ratio = 0.5
                    train_ratio = 0.5
                training_datarows, validation_datarows = train_test_split(datarows, train_size=train_ratio, test_size=validation_ratio)
                datarows = training_datarows
            
            # Serializing json
            json_object = json.dumps(datarows, indent=4)
            # Writing to sample.json
            with open(metadata_path, "w", encoding='utf-8') as outfile:
                outfile.write(json_object)
            
            # Serializing json
            val_json_object = json.dumps(validation_datarows, indent=4)
            # Writing to sample.json
            with open(val_metadata_path, "w", encoding='utf-8') as outfile:
                outfile.write(val_json_object)
                
            # clear memory
            del vae,tokenizer,text_encoder
            gc.collect()
            torch.cuda.empty_cache()
        else:
            with open(metadata_path, "r", encoding='utf-8') as readfile:
                datarows = json.loads(readfile.read())


    # lazy implement of repeats
    datarows_clone = datarows.copy()
    # use epoch rather than repeats for more validation
    repeats = 10
    # repeats is 10, i in range(repeats) would execute 11 times
    for i in range(repeats-1):
        datarows = datarows + datarows_clone.copy()

    del datarows_clone
    torch.set_rng_state(rng_state)
    # ================================================================
    # End create embedding 
    # ================================================================
    
    def collate_fn(datarows):
        # not sure if this would have issue when using multiple aspect ratio
        latents = torch.stack([datarow["latent"].to(accelerator.device) for datarow in datarows])
        prompt_embeds = torch.stack([datarow["prompt_embed"].to(accelerator.device) for datarow in datarows])
        prompt_embed_masks = torch.stack([datarow["prompt_embed_mask"].to(accelerator.device) for datarow in datarows])
        data_infos = default_collate([datarow["data_info"] for datarow in datarows])
        # img_hws = torch.stack([datarow["img_hw"] for datarow in datarows])
        # aspect_ratios = torch.stack([datarow["aspect_ratio"] for datarow in datarows])
        # meta_indexes = default_collate([datarow["meta_index"] for datarow in datarows])
        return {
            "latents": latents,
            "prompt_embeds": prompt_embeds,
            "prompt_embed_masks": prompt_embed_masks,
            "data_infos": data_infos
            # ,
            # "meta_indexes": meta_indexes
            # "img_hws": img_hws,
            # "aspect_ratios": aspect_ratios
        }
    
    # create dataset based on input_dir
    train_dataset = CachedImageDataset(datarows,conditional_dropout_percent=0)

    # referenced from everyDream discord minienglish1 shared script
    #create bucket batch sampler
    bucket_batch_sampler = BucketBatchSampler(train_dataset, batch_size=args.train_batch_size)

    #initialize the DataLoader with the bucket batch sampler
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=bucket_batch_sampler, #use bucket_batch_sampler instead of shuffle
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    
    
    # skip_step = config.skip_step
    total_steps = len(train_dataloader) * args.num_train_epochs

 
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)


    config = dict(
        num_warmup_steps=0,
        lr_schedule = args.lr_scheduler,
        lr_schedule_args = dict(num_warmup_steps=0),
        num_epochs = args.num_train_epochs
    )


    lr_scheduler = build_lr_scheduler(config, optimizer, train_dataloader, 1)
    # lr_scheduler = get_scheduler(
    #     args.lr_scheduler,
    #     optimizer=optimizer,
    #     num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
    #     num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    # )
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_train_steps * args.gradient_accumulation_steps)
    
    # not implement resume_from yet
    if resume_from_path is not None:
        resume_from = dict(
                checkpoint=resume_from_path,
                load_ema=False,
                resume_optimizer=True,
                resume_lr_scheduler=True)
        resume_path = resume_from['checkpoint']
        path = os.path.basename(resume_path)
        start_epoch = int(path.replace('.pth', '').split("_")[1]) - 1
        start_step = int(path.replace('.pth', '').split("_")[3])
        _, missing, unexpected = load_checkpoint(**resume_from,
                                                 model=model,
                                                 optimizer=optimizer,
                                                 lr_scheduler=lr_scheduler,
                                                 max_length=max_length,
                                                 )


    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())


    # prepare everything
    model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        model,optimizer, lr_scheduler, train_dataloader)


    if accelerator.is_main_process:
        # tracker_config = dict(vars(config))
        # tracker_config = config
        # try:
        #     accelerator.init_trackers(args.tracker_project_name, tracker_config)
        # except:
        #     accelerator.init_trackers(f"tb_{timestamp}")
        accelerator.init_trackers("text2image-fine-tune-pixart", config=vars(args))
           
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    # if accelerator.is_main_process:
    #     accelerator.init_trackers("text2image-fine-tune-sdxl", config=vars(args))

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

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    # log_buffer = LogBuffer()
    # time_start, last_tic = time.time(), time.time()
    # print("before epoch start")
    
    for epoch in range(first_epoch, args.num_train_epochs):
        if global_step >= args.max_train_steps:
            break
        data_time_start = time.time()
        data_time_all = 0
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            if global_step < skip_step:
                global_step += 1
                progress_bar.update(1)
                continue    # skip data in the resumed ckpt
            # https://github.com/facebookresearch/schedule_free/blob/main/examples/mnist/main.py#L39
            # optimizer.zero_grad()
            # print("loop dataloader")
            # Sample noise that we'll add to the latents

            grad_norm = None
            data_time_all += time.time() - data_time_start
            with accelerator.accumulate(model):
                if global_step >= args.max_train_steps:
                    break
                optimizer.zero_grad()
                
                with accelerator.autocast():
                    # model_input is vae encoded image aka latent
                    latents = batch["latents"].to(accelerator.device)
                    # get latent like random noise
                    # noise = torch.randn_like(latents)

                    bsz = latents.shape[0]
                    y = batch["prompt_embeds"]
                    y_mask = batch["prompt_embed_masks"]
                    data_info = batch["data_infos"]
                    
                    timesteps = torch.randint(
                        0, train_sampling_steps, (bsz,), device=accelerator.device
                    ).long()
                    # run snr epoch first than back to normal loss
                    # if snr_epoch > 0 and epoch >= snr_epoch:
                    #     train_diffusion.snr = False
                    # else:
                    #     train_diffusion.snr = True
                    loss_term = train_diffusion.training_losses(model, latents, timesteps, model_kwargs=dict(y=y, mask=y_mask, data_info=data_info))
                    loss = loss_term['loss'].mean()
                    
                    
                        
                    #####################################################################
                    # debiased estimation implementation
                    # from kohya ss
                    #####################################################################
                    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                    
                    # def apply_debiased_estimation(loss, timesteps, noise_scheduler):
                    #     snr_t = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])  # batch_size
                    #     snr_t = torch.minimum(snr_t, torch.ones_like(snr_t) * 1000)  # if timestep is 0, snr_t is inf, so limit it to 1000
                    #     weight = 1/torch.sqrt(snr_t)
                    #     loss = weight * loss
                    #     return loss
                    
                    # loss = apply_debiased_estimation(loss, timesteps, train_diffusion)
                    
                    #####################################################################
                    # End debiased estimation implementation section
                    #####################################################################

                    # t = timesteps


                    train_loss += avg_loss.item()
                    
                    # logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                    # # logs = {"step_loss": loss.detach().item(), "lr": args.learning_rate}
                    # progress_bar.set_postfix(**logs)
                    
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(model.parameters(), gradient_clip)
                        if not accelerator.optimizer_step_was_skipped:
                            lr_scheduler.step()

                    optimizer.step()
                    optimizer.zero_grad()
            
            accelerator.wait_for_everyone()
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss / args.gradient_accumulation_steps}, step=global_step)
                train_loss = 0.0
                

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            # logs = {"step_loss": loss.detach().item(), "lr": args.learning_rate}
            progress_bar.set_postfix(**logs)
            # logs.update(lr=lr)
            accelerator.log(logs, step=global_step)

            # progress_bar.update(1)
            # global_step += 1
            # data_time_start = time.time()
            
            # if global_step % save_model_steps == 0:
            #     accelerator.wait_for_everyone()
            #     if accelerator.is_main_process:
            #         os.umask(0o000)
            #         save_checkpoint(os.path.join(args.output_dir, 'checkpoints'),
            #                         epoch=epoch,
            #                         step=global_step,
            #                         model=accelerator.unwrap_model(model).to(dtype=weight_dtype),
            #                         optimizer=optimizer,
            #                         lr_scheduler=lr_scheduler
            #                         )

            # clean memory
            del loss,loss_term,latents,timesteps,y,y_mask,data_info
            gc.collect()
            torch.cuda.empty_cache()

            # not implement yet
        
        
        if global_step < skip_step:
            continue
        
        # store rng before validation
        before_state = torch.random.get_rng_state()
        np_seed = np.random.seed()
        py_state = python_get_rng_state()
        # a = torch.randn(1) 
        if epoch >= skip_epoch and epoch % save_model_epochs == 0 or epoch == args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                os.umask(0o000)
                save_checkpoint(os.path.join(args.output_dir, 'checkpoints'),
                                epoch=epoch,
                                step=global_step,
                                model=accelerator.unwrap_model(model),
                                optimizer=optimizer,
                                lr_scheduler=lr_scheduler
                                )
                
        if epoch % args.validation_epochs == 0 or epoch == args.num_train_epochs - 1:
                # freeze rng
                np.random.seed(0)
                torch.manual_seed(0)
                dataloader_generator = torch.Generator()
                dataloader_generator.manual_seed(0)
                torch.backends.cudnn.deterministic = True
                # b = torch.randn(1) 
                
                # log_validation(model, global_step, device=accelerator.device, vae=vae)
                validation_datarows = []
                with open(val_metadata_path, "r", encoding='utf-8') as readfile:
                    validation_datarows = json.loads(readfile.read())
                
                val_bs = args.train_batch_size
                if args.train_batch_size > len(validation_datarows):
                    val_bs = 1
                # create dataset based on input_dir
                validation_dataset = CachedImageDataset(validation_datarows,conditional_dropout_percent=0)

                # referenced from everyDream discord minienglish1 shared script
                #create bucket batch sampler
                validation_bucket_batch_sampler = BucketBatchSampler(validation_dataset, batch_size=val_bs, drop_last=True)

                #initialize the DataLoader with the bucket batch sampler
                validation_dataloader = torch.utils.data.DataLoader(
                    validation_dataset,
                    batch_sampler=validation_bucket_batch_sampler, #use bucket_batch_sampler instead of shuffle
                    collate_fn=collate_fn,
                    num_workers=args.dataloader_num_workers,
                    generator=dataloader_generator
                )
                
                
                # validation_dataloader = accelerator.prepare(validation_dataloader)
                
                total_loss = 0.0
                num_batches = len(validation_dataloader)
                with torch.cuda.amp.autocast():
                    for step, batch in enumerate(validation_dataloader):
                        # model_input is vae encoded image aka latent
                        latents = batch["latents"].to(accelerator.device)
                        # get latent like random noise
                        # noise = torch.randn_like(latents)

                        bsz = latents.shape[0]
                        y = batch["prompt_embeds"]
                        y_mask = batch["prompt_embed_masks"]
                        data_info = batch["data_infos"]
                        
                        timesteps = torch.randint(
                            0, train_sampling_steps, (bsz,), device=accelerator.device
                        ).long()
                        
                        
                        model_kwargs = dict(data_info=data_info, mask=y_mask)
                        loss_term = train_diffusion.training_losses(model, latents, timesteps, model_kwargs=dict(y=y, data_info=data_info, mask=y_mask))
                        total_loss = loss_term['loss'].mean()
                        
                        del loss_term, latents, timesteps, y, y_mask, data_info
                        gc.collect()
                        torch.cuda.empty_cache()
                        
                        
                
                avg_loss = total_loss / num_batches
                logs = {"val_loss": avg_loss, "val_lr": lr_scheduler.get_last_lr()[0]}
                # logs = {"step_loss": loss.detach().item(), "lr": args.learning_rate}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                del validation_datarows, validation_dataset, validation_bucket_batch_sampler, validation_dataloader
                gc.collect()
                torch.cuda.empty_cache()
            
        # restore rng before validation
        np.random.seed(np_seed)
        torch.random.set_rng_state(before_state)
        torch.backends.cudnn.deterministic = False
        version, state, gauss = py_state
        python_set_rng_state((version, tuple(state), gauss))
        
        # clear memory
        del before_state,np_seed,py_state,version,state,gauss
        gc.collect()
        torch.cuda.empty_cache()
        # c = torch.randn(1) 
        # print("rng test:",a,b,c)
        accelerator.wait_for_everyone()
        
        # stop training in earlier epoch
        if break_epoch !=0 and epoch >= break_epoch:
            break
    
    # ==================================================
    # validation part after training
    # ==================================================
    if accelerator.is_main_process:
        # save model
        # save_path = os.path.join(args.output_dir, f"{args.save_name}-{global_step}")
        # accelerator.save_state(save_path)

        # del pipeline,noise_scheduler
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