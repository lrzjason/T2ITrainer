
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
# import logging
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
from diffusers.utils.torch_utils import is_compiled_module

from diffusers.training_utils import cast_training_params

from utils.image_utils_sd3 import CachedImageDataset, create_metadata_cache
from utils.bucket.bucket_batch_sampler import BucketBatchSampler


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
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    # parser.add_argument(
    #     "--resume_from_checkpoint",
    #     type=str,
    #     default=None,
    #     help=(
    #         "Whether training should be resumed from a previous checkpoint. Use a path saved by"
    #         ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
    #     ),
    # )
    
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

    # parser.add_argument(
    #     "--use_8bit_adam",
    #     action="store_true",
    #     help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    # )

    # parser.add_argument(
    #     "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    # )
    # parser.add_argument(
    #     "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    # )
    # parser.add_argument(
    #     "--prodigy_beta3",
    #     type=float,
    #     default=None,
    #     help="coefficients for computing the Prodidy stepsize using running averages. If set to None, "
    #     "uses the value of square root of beta2. Ignored if optimizer is adamW",
    # )
    # parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    # parser.add_argument("--adam_weight_decay", type=float, default=1e-02, help="Weight decay to use for unet params")
    # parser.add_argument(
    #     "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    # )

    # parser.add_argument(
    #     "--adam_epsilon",
    #     type=float,
    #     default=1e-08,
    #     help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    # )

    # parser.add_argument(
    #     "--prodigy_use_bias_correction",
    #     type=bool,
    #     default=True,
    #     help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    # )
    # parser.add_argument(
    #     "--prodigy_safeguard_warmup",
    #     type=bool,
    #     default=True,
    #     help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
    #     "Ignored if optimizer is adamW",
    # )
    # parser.add_argument(
    #     "--prodigy_d_coef",
    #     type=float,
    #     default=2,
    #     help=("The dimension of the LoRA update matrices."),
    # )
    # parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
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


def load_text_encoders(class_one, class_two, class_three,revision=None,variant=None):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=revision, variant=variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=revision, variant=variant
    )
    text_encoder_three = class_three.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_3", revision=revision, variant=variant
    )
    return text_encoder_one, text_encoder_two, text_encoder_three

def main(args):
    lr_num_cycles = 1
    lr_power = 1
    resume_from_checkpoint = ""
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
    # args.validation_prompt = ""
    
    args.seed = 4321
    args.logging_dir = 'logs'
    args.mixed_precision = "fp16"
    args.report_to = "wandb"
    args.lr_warmup_steps = 1
    args.lr_scheduler = "cosine"
    args.output_dir = 'F:/models/sd3'
    args.save_name = "sd3_cinematic"
    args.train_data_dir = "F:/ImageSet/handpick_high_quality_b2_train"
    args.learning_rate = 1
    args.train_batch_size = 1
    args.repeats = 10
    args.gradient_accumulation_steps = 10
    args.num_train_epochs = 30
    args.validation_epochs = 1
    args.rank = 32
    args.save_model_epochs = 1
    args.skip_epoch = 1
    args.break_epoch = 0
    args.skip_step = 0
    args.gradient_checkpointing = True
    args.validation_ratio = 0.1
    args.num_validation_images = 1
    args.pretrained_model_name_or_path = "stabilityai/stable-diffusion-3-medium-diffusers"
    args.model_path = None # "F:/models/Stable-diffusion/sd3/opensd3.safetensors"
    args.optimizer = "prodigy"
    
    
    # create metadata.jsonl if not exist
    metadata_path = os.path.join(args.train_data_dir, 'metadata_sd3.json')
    val_metadata_path =  os.path.join(args.train_data_dir, 'val_metadata_sd3.json')
    
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
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    
    offload_device = accelerator.device
    
    if not os.path.exists(metadata_path) or not os.path.exists(val_metadata_path):
        offload_device = torch.device("cpu")
    
    if args.model_path is None:
        transformer = SD3Transformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="transformer", revision=revision, variant=variant
        ).to(offload_device, dtype=weight_dtype)
    else:
        transformer = SD3Transformer2DModel.from_single_file(args.model_path)

    transformer.requires_grad_(False)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # now we will add new LoRA weights to the attention layers
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer.add_adapter(transformer_lora_config)
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
    



    # ==========================================================
    # Create train dataset
    # ==========================================================
    # data_files = {}
    # this part need more work afterward, you need to prepare 
    # the train files and val files split first before the training
    if args.train_data_dir is not None:
        # data_files["train"] = os.path.join(args.train_data_dir, "**")
        
        # Load the tokenizers
        tokenizer_one = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=revision,
        )
        tokenizer_two = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            revision=revision,
        )
        tokenizer_three = T5TokenizerFast.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer_3",
            revision=revision,
        )

        # import correct text encoder classes
        text_encoder_cls_one = import_model_class_from_model_name_or_path(
            args.pretrained_model_name_or_path, revision
        )
        text_encoder_cls_two = import_model_class_from_model_name_or_path(
            args.pretrained_model_name_or_path, revision, subfolder="text_encoder_2"
        )
        text_encoder_cls_three = import_model_class_from_model_name_or_path(
            args.pretrained_model_name_or_path, revision, subfolder="text_encoder_3"
        )

        text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
            text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three
        )
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=revision,
            variant=variant,
        )
        
        vae.requires_grad_(False)
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
        text_encoder_three.requires_grad_(False)
        
        vae.to(accelerator.device, dtype=torch.float32)
        text_encoder_one.to(accelerator.device, dtype=weight_dtype)
        text_encoder_two.to(accelerator.device, dtype=weight_dtype)
        text_encoder_three.to(accelerator.device, dtype=weight_dtype)

        
        tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
        text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]
        
        

        
        datarows = []
        if not os.path.exists(metadata_path) or not os.path.exists(val_metadata_path):
            # using compel for longer prompt embedding
            # compel = Compel(tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2] , text_encoder=[text_encoder_one, text_encoder_two], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])

            # create metadata and latent cache
            datarows = create_metadata_cache(tokenizers,text_encoders,vae,args.train_data_dir,repeats=args.repeats)
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
            del validation_datarows, vae,tokenizer_one, tokenizer_two, tokenizer_three, text_encoder_one, text_encoder_two, text_encoder_three
            gc.collect()
            torch.cuda.empty_cache()
        else:
            with open(metadata_path, "r", encoding='utf-8') as readfile:
                datarows = json.loads(readfile.read())


    # resume from cpu after cache files
    transformer.to(accelerator.device)

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
    # def collate_npz_path_fn(examples):
    #     npz_paths = [example["npz_path"] for example in examples]
    #     return {
    #         "npz_paths":npz_paths
    #     }
    
    
    # create dataset based on input_dir
    train_dataset = CachedImageDataset(datarows,conditional_dropout_percent=0.1)

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
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0


    # Potentially load in the weights and states from a previous save
    if resume_from_checkpoint:
        if resume_from_checkpoint != "latest":
            path = os.path.basename(resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
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
                

                # Backpropagate
                accelerator.backward(loss)
                step_loss = loss.detach().item()
                del loss
                if accelerator.sync_gradients:
                    params_to_clip = transformer_lora_parameters
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
                
                lr = lr_scheduler.get_last_lr()[0]
                lr_name = "lr"
                if args.optimizer == "prodigy":
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
        np_seed = np.random.seed()
        py_state = python_get_rng_state()
        
        if accelerator.is_main_process:
            if epoch >= args.skip_epoch and epoch % args.save_model_epochs == 0 or epoch == args.num_train_epochs - 1:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"{args.save_name}-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
                    
            if epoch % args.validation_epochs == 0 and epoch > 0:
                with torch.no_grad():
                    transformer = unwrap_model(transformer)
                    # freeze rng
                    np.random.seed(0)
                    torch.manual_seed(0)
                    dataloader_generator = torch.Generator()
                    dataloader_generator.manual_seed(0)
                    torch.backends.cudnn.deterministic = True
                    
                    validation_datarows = []
                    with open(val_metadata_path, "r", encoding='utf-8') as readfile:
                        validation_datarows = json.loads(readfile.read())
                    
                    if len(validation_datarows)>0:
                        validation_dataset = CachedImageDataset(validation_datarows,conditional_dropout_percent=0)
                        
                        val_batch_sampler = BucketBatchSampler(validation_dataset, batch_size=args.train_batch_size)

                        #initialize the DataLoader with the bucket batch sampler
                        val_dataloader = torch.utils.data.DataLoader(
                            validation_dataset,
                            batch_sampler=val_batch_sampler, #use bucket_batch_sampler instead of shuffle
                            collate_fn=collate_fn,
                            num_workers=dataloader_num_workers,
                        )

                        print("beginning loss_validation")
                        
                        total_loss = 0.0
                        num_batches = len(val_dataloader)
                        # basically the as same as the training loop
                        for i, batch in enumerate(val_dataloader):
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

                            
                            model_pred = transformer(
                                hidden_states=noisy_model_input,
                                timestep=timesteps,
                                encoder_hidden_states=prompt_embeds,
                                pooled_projections=pooled_prompt_embeds,
                                return_dict=False,
                            )[0]
                            model_pred.to(device=latents.device,dtype=weight_dtype)
                            
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
                            total_loss += loss
                            del loss, target, weighting, model_pred, noisy_model_input, sigmas, timesteps, indices, bsz, noise
                            del latents, prompt_embeds, pooled_prompt_embeds
                            gc.collect()
                            torch.cuda.empty_cache()
                            
                        avg_loss = total_loss / num_batches
                        
                        lr = lr_scheduler.get_last_lr()[0]
                        lr_name = "val_lr"
                        if args.optimizer == "prodigy":
                            lr = lr_scheduler.optimizers[-1].param_groups[0]["d"] * lr_scheduler.optimizers[-1].param_groups[0]["lr"]
                            lr_name = "val_lr lr/d*lr"
                        logs = {"val_loss": avg_loss, lr_name: lr, "epoch": epoch}
                        progress_bar.set_postfix(**logs)
                        accelerator.log(logs, step=global_step)
                        del num_batches, avg_loss, total_loss, validation_datarows, validation_dataset, 
                        del val_batch_sampler, val_dataloader
                        gc.collect()
                        torch.cuda.empty_cache()
            
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

if __name__ == "__main__":
    args = parse_args()
    main(args)