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
# 20240710 add kolors training, dir kolors copied from https://github.com/Kwai-Kolors/Kolors
from diffusers.models.attention_processor import AttnProcessor2_0
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
from utils.image_utils_kolors import BucketBatchSampler, CachedImageDataset, create_metadata_cache

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

from hashlib import md5
import glob

from PIL import Image
import os, torch
# from PIL import Image
from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256 import StableDiffusionXLPipeline
from kolors.models.modeling_chatglm import ChatGLMModel
from kolors.models.tokenization_chatglm import ChatGLMTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers import EulerDiscreteScheduler

from utils.image_utils_kolors import compute_text_embeddings
from utils.dist_utils import flush

from utils.utils import get_md5_by_path


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.30.0.dev0")

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
        "--train_data_dir",
        type=str,
        default="",
        help=(
            "output folder for the slider training"
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
    parser.add_argument(
        "--main_prompt",
        type=str,
        default="a girl",
        help=(
            "the main prompt for both positive images and negative images"
        ),
    )
    parser.add_argument(
        "--uncondition_prompt",
        type=str,
        default="abstruct",
        help=(
            "the main uncondition prompt for both positive images and negative images"
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
        type=int,
        default=5,
        help=(
            "Image generation guidance_scale"
        ),
    )
    parser.add_argument(
        "--generation_batch",
        type=int,
        default=10,
        help=(
            "Total batch of generation"
        ),
    )
    # parser.add_argument(
    #     "--batch_size",
    #     type=int,
    #     default=1,
    #     help=(
    #         "batch size of generation"
    #     ),
    # )
    
    parser.add_argument("--seed", type=int, default=None, help="A seed for generation init.")
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

@torch.no_grad()
def main(args):
    args.train_data_dir = "F:/ImageSet/kolors_slider"
    args.image_prefix = "sky"
    args.main_prompt = "photo of sky"
    args.uncondition_prompt = "star, starry, oil painting"
    args.pos_prompt = "clean blue sky, day light"
    args.neg_prompt = "chaotic dark sky, dark night"
    # args.batch_size = 2
    args.generation_batch = 5
    args.pretrained_model_name_or_path = "F:/T2ITrainer/kolors_models"
    args.steps = 30
    args.cfg = 3.5
    args.seed = 1
    
    main_prompt = args.main_prompt
    uncondition_prompt = args.uncondition_prompt
    pos_prompt = f'{main_prompt}, {args.pos_prompt}'
    neg_prompt = f'{main_prompt}, {args.neg_prompt}'
    # batch_size = args.batch_size
    generation_batch = args.generation_batch
    ckpt_dir = args.pretrained_model_name_or_path
    steps = args.steps
    cfg = args.cfg
    
    # fix seed
    seed = args.seed
    
    
    metadata_file = "metadata_kolors_slider.json"
    metadata_path = os.path.join(args.train_data_dir, metadata_file)
    metadata = {
        'main_prompt':main_prompt,
        'uncondition_prompt':uncondition_prompt,
        'pos_prompt':pos_prompt,
        'neg_prompt':neg_prompt,
        'generation_batch':generation_batch,
        'pretrained_model_name_or_path':ckpt_dir,
        'steps':steps,
        'cfg':cfg,
        'seed':seed,
    }
    
    # freeze rng
    np.random.seed(seed)
    torch.manual_seed(seed)
    resolution = 1024
    
    text_encoder = ChatGLMModel.from_pretrained(
        f'{ckpt_dir}/text_encoder',
        torch_dtype=torch.float16).half()
    tokenizer = ChatGLMTokenizer.from_pretrained(f'{ckpt_dir}/text_encoder')
    vae = AutoencoderKL.from_pretrained(f"{ckpt_dir}/vae").half()
    scheduler = EulerDiscreteScheduler.from_pretrained(f"{ckpt_dir}/scheduler")
    unet = UNet2DConditionModel.from_pretrained(f"{ckpt_dir}/unet").half()
    # pipe = OriStableDiffusionXLPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-xl-base-1.0", 
    #     torch_dtype=torch.float16
    # )
    pipe = StableDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        force_zeros_for_empty_prompt=False
    ).to("cuda")
    pipe.enable_model_cpu_offload()
    
    prompt_embeds_list = []
    metadata['generation_configs'] = [
        {
            'set_name':"positive",
            'prompt': pos_prompt,
        },
        {
            
            'set_name':"negative",
            'prompt': neg_prompt,
        },
        {
            
            'set_name':"uncondition",
            'prompt': uncondition_prompt,
        },
    ]
    for generation_config in metadata['generation_configs']:
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
        npz_path = f"{args.train_data_dir}/{set_name}.npkolors"
        # save latent to cache file
        torch.save(npz_dict, npz_path)
        generation_config['npz_path'] = npz_path
        npz_path_md5 = get_md5_by_path(npz_path)
        generation_config['npz_path_md5'] = npz_path_md5
        prompt_embeds_list.append((set_name,prompt_embeds, pooled_prompt_embeds))
    
    _, uncondition_prompt_embeds, uncondition_pooled_prompt_embeds = prompt_embeds_list.pop()
    
    del text_encoder, tokenizer
    flush()
    for j in range(len(prompt_embeds_list)):
        # align seed with positive and negative
        sample_seed = seed
        set_name, prompt_embeds, pooled_prompt_embeds = prompt_embeds_list[j]
        save_dir = f"{args.train_data_dir}/{set_name}"
        metadata['generation_configs'][j]['item_list'] = []
        os.makedirs(save_dir, exist_ok=True)
        for i in range(generation_batch):
            output,latent = pipe(
                prompt_embeds=prompt_embeds, 
                pooled_prompt_embeds=pooled_prompt_embeds, 
                negative_prompt_embeds=uncondition_prompt_embeds,
                negative_pooled_prompt_embeds=uncondition_pooled_prompt_embeds,
                height=resolution,
                width=resolution,
                num_inference_steps=steps,
                guidance_scale=cfg,
                num_images_per_prompt=1,
                generator= torch.Generator(pipe.device).manual_seed(sample_seed),
                )
            # save latent
            latent_path = f"{save_dir}/{args.image_prefix}_{sample_seed}.nplatent"
            time_id = torch.tensor(list((1024, 1024) + 
                                        (0,0) + 
                                        (1024, 1024))).to(vae.device, dtype=vae.dtype)
            latent_dict = {
                'latent': latent[0].cpu(),
                'time_id': time_id.cpu(),
            }
            torch.save(latent_dict, latent_path)
            
            # save image
            image = output.images[0]
            image_path = f"{save_dir}/{args.image_prefix}_{sample_seed}.webp"
            image.save(image_path)
            
            training_item = {
                'bucket': "1024x1024",
                'latent_path':latent_path,
                'latent_path_md5':get_md5_by_path(latent_path),
                'image_path':image_path,
                'image_path_md5':get_md5_by_path(image_path),
            }
            metadata['generation_configs'][j]['item_list'].append(training_item)
            sample_seed += 1
            
    # save metadata
    with open(metadata_path, "w", encoding='utf-8') as writefile:
        writefile.write(json.dumps(metadata, indent=4))


if __name__ == "__main__":
    args = parse_args()
    main(args)