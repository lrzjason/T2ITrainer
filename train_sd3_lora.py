
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

import argparse
import copy
import gc
import itertools
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, PretrainedConfig, T5EncoderModel, T5TokenizerFast

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.training_utils import cast_training_params
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module


import sys
sys.path.append('F:/T2ITrainer/utils')
import image_utils_sd3
from image_utils_sd3 import BucketBatchSampler, CachedImageDataset


from sklearn.model_selection import train_test_split


import json

from prodigyopt import Prodigy

# https://github.com/Lightning-AI/pytorch-lightning/blob/0d52f4577310b5a1624bed4d23d49e37fb05af9e/src/lightning_fabric/utilities/seed.py
from random import getstate as python_get_rng_state
from random import setstate as python_set_rng_state


from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")

logger = get_logger(__name__)



def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
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
        "--instance_data_dir",
        type=str,
        default=None,
        help=("A folder containing the training data. "),
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column of the dataset containing the instance prompt for each image",
    )

    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")

    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=False,
        help="The prompt with identifier specifying the instance, e.g. 'photo of a TOK dog', 'in the style of TOK'",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
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
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
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
        "--resolution",
        type=int,
        default=512,
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
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
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
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
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

    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
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
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme", type=str, default="sigma_sqrt", choices=["sigma_sqrt", "logit_normal", "mode"]
    )
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
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
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
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
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
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
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
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
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    # if args.dataset_name is None and args.instance_data_dir is None:
    #     raise ValueError("Specify either `--dataset_name` or `--instance_data_dir`")

    # if args.dataset_name is not None and args.instance_data_dir is not None:
    #     raise ValueError("Specify only one of `--dataset_name` or `--instance_data_dir`")

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    return args


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
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


def load_text_encoders(class_one, class_two, class_three):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    text_encoder_three = class_three.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_3", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two, text_encoder_three

def main(args):
    args.seed = 4321
    args.output_dir = 'F:/models/sd3/output'
    args.logging_dir = 'logs'
    args.model_path = 'F:/models/Stable-diffusion/sdxl/o2/openxl2_008.safetensors'
    args.mixed_precision = "bf16"
    # args.report_to = "tensorboard"
    
    args.report_to = "wandb"
    args.enable_xformers_memory_efficient_attention = True
    args.gradient_checkpointing = False
    args.allow_tf32 = True


    # args.train_data_dir = 'F:/ImageSet/openxl2_realism'
    # try to use clip filtered dataset
    # args.train_data_dir = 'F:/ImageSet/openxl2_realism_above_average'
    # args.train_data_dir = 'F:/ImageSet/openxl2_realism_above_average'
    # args.num_train_epochs = 5
    args.lr_warmup_steps = 1
    args.lr_num_cycles = 1
    args.lr_power = 1
    
    
    # reduce lr from 1e-5 to 2e-6
    # args.learning_rate = 1.2e-6
    args.learning_rate = 1
    args.train_batch_size = 1
    # reduce gas from 500 to 100
    args.gradient_accumulation_steps = 1
    # increase save steps from 50 to 250
    args.checkpointing_steps = 90
    args.resume_from_checkpoint = ""
    # args.resume_from_checkpoint = "F:/models/unet/output/actual_run-50"
    args.save_name = "sd3_test"
    # args.lr_scheduler = "constant"
    args.lr_scheduler = "cosine"

    
    # args.train_data_dir = 'F:/ImageSet/openxl2_realism_test'
    # args.resume_from_checkpoint = "F:/models/unet/output/test_run-50"
    # args.num_train_epochs = 2
    # args.train_batch_size = 1
    # args.gradient_accumulation_steps = 1
    # args.save_name = "test_run"

    args.scale_lr = False
    args.use_8bit_adam = True
    args.adam_beta1 = 0.9
    # args.adam_beta2 = 0.999
    args.adam_beta2 = 0.99

    args.adam_weight_decay = 1e-2
    args.adam_epsilon = 1e-08
    # args.train_data_dir = 'F:/ImageSet/training_script_testset_sdxl_train_validation/train/test'
    # args.train_data_dir = 'F:/ImageSet/training_script_cotton_doll/train'
    # args.train_data_dir = 'F:/ImageSet/openxl2_realism_test'
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
    args.validation_prompt = ""
    
    # args.validation_epochs = 1
    
    args.train_data_dir = "F:/ImageSet/pixart_test_cropped"
    # args.train_data_dir = "F:/ImageSet/pixart_test_one"
    # args.train_data_dir = "F:/ImageSet/openxl2_reg_test"
    # args.train_data_dir = "F:/ImageSet/dog"
    args.num_train_epochs = 300
    save_model_epochs = 10
    args.validation_epochs = 10
    args.rank = 8
    skip_epoch = 0
    break_epoch = 0
    skip_step = 0
    
    args.validation_ratio = 0.1
    args.num_validation_images = 1
    args.caption_exts = '.txt,.wd14_cap'

    # vae_path = "F:/models/VAE/sdxl_vae.safetensors"
    # clip_l_path = "F:/models/clip/clip_l.safetensors"
    # clip_g_path = "F:/models/clip/clip_g.safetensors"
    # t5xxl_path = "F:/models/clip/t5xxl_fp16.safetensors"
    
    # stabilityai/stable-diffusion-3-medium-diffusers
    args.pretrained_model_name_or_path = "stabilityai/stable-diffusion-3-medium-diffusers"
    
    
    args.revision = None
    args.variant = None
    
    args.optimizer = "prodigy"
    args.prodigy_decouple = True
    args.prodigy_beta3 = None
    args.prodigy_use_bias_correction = True
    args.prodigy_safeguard_warmup = True
    
    # accelerator = {
    #     "device": "cuda" if torch.cuda.is_available() else "cpu",
    #     "mixed_precision": "bf16"
    # }
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
    

    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_3",
        revision=args.revision,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_3"
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
        text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    ).to(accelerator.device, dtype=weight_dtype)

    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)

    # if args.gradient_checkpointing:
    #     transformer.enable_gradient_checkpointing()

    # now we will add new LoRA weights to the attention layers
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer.add_adapter(transformer_lora_config)



    # if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
    #     # due to pytorch#99272, MPS does not yet support bfloat16.
    #     raise ValueError(
    #         "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
    #     )

    vae.to(accelerator.device, dtype=torch.float32)
    # not train_text_encoder
    # if not args.train_text_encoder:
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    text_encoder_three.to(accelerator.device, dtype=weight_dtype)

    # if args.gradient_checkpointing:
    #     transformer.enable_gradient_checkpointing()
        # if args.train_text_encoder:
        #     text_encoder_one.gradient_checkpointing_enable()
        #     text_encoder_two.gradient_checkpointing_enable()
        #     text_encoder_three.gradient_checkpointing_enable()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            StableDiffusion3Pipeline.save_lora_weights(
                output_dir, transformer_lora_layers=transformer_lora_layers_to_save
            )

    def load_model_hook(models, input_dir):
        transformer_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")
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


    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    # if args.allow_tf32 and torch.cuda.is_available():
    #     torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
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
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
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
        # if args.train_text_encoder and args.text_encoder_lr:
        #     logger.warning(
        #         f"Learning rates were provided both for the transformer and the text encoder- e.g. text_encoder_lr:"
        #         f" {args.text_encoder_lr} and learning_rate: {args.learning_rate}. "
        #         f"When using prodigy only learning_rate is used as the initial learning rate."
        #     )
        #     # changes the learning rate of text_encoder_parameters_one and text_encoder_parameters_two to be
        #     # --learning_rate
        #     params_to_optimize[1]["lr"] = args.learning_rate
        #     params_to_optimize[2]["lr"] = args.learning_rate
        #     params_to_optimize[3]["lr"] = args.learning_rate

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )
    

    tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
    text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]


    # ==========================================================
    # Create train dataset
    # ==========================================================
    # data_files = {}
    # this part need more work afterward, you need to prepare 
    # the train files and val files split first before the training
    if args.train_data_dir is not None:
        # data_files["train"] = os.path.join(args.train_data_dir, "**")
        
        datarows = []
        # create metadata.jsonl if not exist
        metadata_path = os.path.join(args.train_data_dir, 'metadata_sd3.json')
        val_metadata_path =  os.path.join(args.train_data_dir, 'val_metadata_sd3.json')
        if not os.path.exists(metadata_path) or not os.path.exists(val_metadata_path):
            # using compel for longer prompt embedding
            # compel = Compel(tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2] , text_encoder=[text_encoder_one, text_encoder_two], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])

            # create metadata and latent cache
            datarows = image_utils_sd3.create_metadata_cache(tokenizers,text_encoders,vae,args.train_data_dir)
            # Serializing json
            json_object = json.dumps(datarows, indent=4)
            # Writing to sample.json
            with open(metadata_path, "w", encoding='utf-8') as outfile:
                outfile.write(json_object)
            
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
            
            if len(validation_datarows) > 0:
                # Serializing json
                val_json_object = json.dumps(validation_datarows, indent=4)
                # Writing to sample.json
                with open(val_metadata_path, "w", encoding='utf-8') as outfile:
                    outfile.write(val_json_object)
                
            # clear memory
            del vae,tokenizer_one, tokenizer_two, tokenizer_three, text_encoder_one, text_encoder_two, text_encoder_three
            gc.collect()
            torch.cuda.empty_cache()
        else:
            with open(metadata_path, "r", encoding='utf-8') as readfile:
                datarows = json.loads(readfile.read())



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
    train_dataset = CachedImageDataset(datarows,conditional_dropout_percent=0.1)

    # referenced from everyDream discord minienglish1 shared script
    #create bucket batch sampler
    bucket_batch_sampler = BucketBatchSampler(train_dataset, batch_size=args.train_batch_size, drop_last=True)

    #initialize the DataLoader with the bucket batch sampler
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=bucket_batch_sampler, #use bucket_batch_sampler instead of shuffle
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )
    
    

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



    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )


    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "dreambooth-sd3-lora"
        accelerator.init_trackers(tracker_name, config=vars(args))
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


    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
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
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0
    # initial_global_step = 0



    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    
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
        # train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # models_to_accumulate = [transformer]
            # https://github.com/facebookresearch/schedule_free/blob/main/examples/mnist/main.py#L39
            # optimizer.zero_grad()
            # print("loop dataloader")
            with accelerator.accumulate(transformer):
                latents = batch["latents"].to(accelerator.device)
                prompt_embeds = batch["prompt_embeds"].to(accelerator.device)
                pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(accelerator.device)
                # get latent like random noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # Sample a random timestep for each image
                indices = torch.randint(0, noise_scheduler_copy.config.num_train_timesteps, (bsz,))
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=latents.device)

                # Add noise according to flow matching.
                sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                noisy_model_input = sigmas * noise + (1.0 - sigmas) * latents

                with accelerator.autocast():
                    model_pred = transformer(
                        hidden_states=noisy_model_input,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        return_dict=False,
                    )[0]
                    
                    # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                    # Preconditioning of the model outputs.
                    model_pred = model_pred * (-sigmas) + noisy_model_input

                    # for simple implementation only use weighting_scheme sigma_sqrt
                    # which is the default option
                    weighting = (sigmas**-2.0).float()
                    
                    
                    # simplified flow matching aka 0-rectified flow matching loss
                    # target = model_input - noise
                    target = latents
                    # Compute regular loss.
                    loss = torch.mean(
                        (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                        1,
                    )
                    loss = loss.mean()
                    assert loss.dtype is torch.float32
                
                
                del target, weighting,model_pred, noisy_model_input, sigmas, timesteps, indices, bsz, noise
                del latents, prompt_embeds, pooled_prompt_embeds
                

                # Gather the losses across all processes for logging (if we use distributed training).
                # avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                # # train_loss += avg_loss.item() / args.gradient_accumulation_steps
                # # args.gradient_accumulation_steps
                # # due to 'with accelerator.accumulate(unet):' doesn't need to handle ' / args.gradient_accumulation_steps'
                # # https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation 
                # train_loss += avg_loss.item()

                
                
                # Backpropagate
                accelerator.backward(loss)
                step_loss = loss.detach().item()
                del loss
                if accelerator.sync_gradients:
                    params_to_clip = transformer_lora_parameters
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                #post batch check for gradient updates
                accelerator.wait_for_everyone()
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    # accelerator.log({"train_loss": train_loss / args.gradient_accumulation_steps}, step=global_step)
                    # train_loss = 0.0
                    
                    # if accelerator.is_main_process:
                    #     if global_step % args.checkpointing_steps == 0:
                    #         save_path = os.path.join(args.output_dir, f"{args.save_name}-{global_step}")
                    #         accelerator.save_state(save_path)
                    #         logger.info(f"Saved state to {save_path}")

                
                logs = {"step_loss": step_loss, "lr": lr_scheduler.get_last_lr()[0]}
                accelerator.log(logs, step=global_step)
                # logs = {"step_loss": loss.detach().item(), "lr": args.learning_rate}
                progress_bar.set_postfix(**logs)
                
                if global_step >= args.max_train_steps:
                    break
                
                gc.collect()
                torch.cuda.empty_cache()
            
        # ==================================================
        # validation part
        # ==================================================
        
        if global_step < skip_step:
            continue
        
        # store rng before validation
        # before_state = torch.random.get_rng_state()
        # np_seed = np.random.seed()
        # py_state = python_get_rng_state()
        
        if accelerator.is_main_process:
            if epoch >= skip_epoch and epoch % save_model_epochs == 0 or epoch == args.num_train_epochs - 1:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    # transformer = unwrap_model(transformer)
                    save_path = os.path.join(args.output_dir, f"{args.save_name}-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
                    
        #     if epoch % args.validation_epochs == 0 and epoch > 0:
        #         # freeze rng
        #         np.random.seed(0)
        #         torch.manual_seed(0)
        #         dataloader_generator = torch.Generator()
        #         dataloader_generator.manual_seed(0)
        #         torch.backends.cudnn.deterministic = True
                
        #         print('epoch',epoch)
        #         print('args.validation_epochs',args.validation_epochs)
        #         print('epoch _ args.validation_epochs',epoch % args.validation_epochs)
                
        #         validation_datarows = []
        #         with open(val_metadata_path, "r", encoding='utf-8') as readfile:
        #             validation_datarows = json.loads(readfile.read())
                
        #         val_bs = args.train_batch_size
        #         if args.train_batch_size > len(validation_datarows):
        #             val_bs = 1
        #         if len(validation_datarows)>0:
        #             validation_dataset = CachedImageDataset(validation_datarows,conditional_dropout_percent=0)
                    
        #             val_batch_sampler = BucketBatchSampler(validation_dataset, batch_size=args.train_batch_size, drop_last=True)

        #             #initialize the DataLoader with the bucket batch sampler
        #             val_dataloader = torch.utils.data.DataLoader(
        #                 validation_dataset,
        #                 batch_sampler=val_batch_sampler, #use bucket_batch_sampler instead of shuffle
        #                 collate_fn=collate_fn,
        #                 num_workers=args.dataloader_num_workers,
        #             )

        #             print("beginning loss_validation")
                    
        #             total_loss = 0.0
        #             num_batches = len(val_dataloader)
        #             with torch.cuda.amp.autocast():
        #                 # basically the as same as the training loop
        #                 for i, batch in enumerate(val_dataloader):
        #                     latents = batch["latents"].to(accelerator.device)
        #                     prompt_embeds = batch["prompt_embeds"].to(accelerator.device)
        #                     pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(accelerator.device)
        #                     # get latent like random noise
        #                     noise = torch.randn_like(latents)
        #                     bsz = latents.shape[0]
                            
        #                     # Sample a random timestep for each image
        #                     indices = torch.randint(0, noise_scheduler_copy.config.num_train_timesteps, (bsz,))
        #                     timesteps = noise_scheduler_copy.timesteps[indices].to(device=latents.device)

        #                     # Add noise according to flow matching.
        #                     sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
        #                     noisy_model_input = sigmas * noise + (1.0 - sigmas) * latents

        #                     model_pred = transformer(
        #                         hidden_states=noisy_model_input,
        #                         timestep=timesteps,
        #                         encoder_hidden_states=prompt_embeds,
        #                         pooled_projections=pooled_prompt_embeds,
        #                         return_dict=False,
        #                     )[0]
                            
        #                     # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
        #                     # Preconditioning of the model outputs.
        #                     model_pred = model_pred * (-sigmas) + noisy_model_input

        #                     # for simple implementation only use weighting_scheme sigma_sqrt
        #                     # which is the default option
        #                     weighting = (sigmas**-2.0).float()
                            
                            
        #                     # simplified flow matching aka 0-rectified flow matching loss
        #                     # target = model_input - noise
        #                     target = latents
        #                     # Compute regular loss.
        #                     loss = torch.mean(
        #                         (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
        #                         1,
        #                     )
        #                     loss = loss.mean()
        #                     total_loss += loss
        #                     del target, weighting,model_pred, noisy_model_input, sigmas, timesteps, indices, bsz, noise
        #                     del latents, prompt_embeds, pooled_prompt_embeds
        #                     gc.collect()
        #                     torch.cuda.empty_cache()
        #                     break
                        
        #             avg_loss = total_loss / num_batches
        #             logs = {"val_loss": avg_loss, "val_lr": lr_scheduler.get_last_lr()[0]}
        #             # logs = {"step_loss": loss.detach().item(), "lr": args.learning_rate}
        #             progress_bar.set_postfix(**logs)
        #             accelerator.log(logs, step=global_step)
        #             del validation_datarows, validation_dataset, val_batch_sampler, val_dataloader
        #             gc.collect()
        #             torch.cuda.empty_cache()
        
        # # restore rng before validation
        # np.random.seed(np_seed)
        # torch.random.set_rng_state(before_state)
        # torch.backends.cudnn.deterministic = False
        # version, state, gauss = py_state
        # python_set_rng_state((version, tuple(state), gauss))
        
        # ==================================================
        # end validation part
        # ==================================================
    
    accelerator.end_training()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)