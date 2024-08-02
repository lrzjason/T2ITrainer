# Thanks to Kwai/kolors for the open source weight and comfyanonymous/ComfyUI for unet mapping code

import torch
import safetensors
from safetensors import safe_open
from safetensors.torch import save_file
import json
import comfy.utils as utils
import copy
import argparse

def main(args):
    print("running merge_state_dict")
    # kolors_model_path = "F:/models/unet/new_kolors/diffusion_pytorch_model.fp16.safetensors"
    # sdxl_model_path = "F:/models/Stable-diffusion/sdxl/o2/openxl2_030.safetensors"
    # merged_kolors_path = "F:/Kolors/unet/diffusion_pytorch_model.merge_with_perturbed.fp16.safetensors"
    # ratio = 0.25
    # perturbed_ratio = 0.02
    kolors_model_path = args.kolors_model_path
    sdxl_model_path = args.sdxl_model_path
    merged_kolors_path = args.merged_kolors_path
    ratio = args.ratio
    perturbed_ratio = args.perturbed_ratio
    print("kolors_model_path:",kolors_model_path)
    print("sdxl_model_path:",sdxl_model_path)


    kolors_model = safetensors.safe_open(kolors_model_path, 'pt')
    keys = kolors_model.keys()
    model_a = {key:kolors_model.get_tensor(key) for key in keys}
    model_b = safetensors.torch.load_file(sdxl_model_path)
    SDXL = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
                'num_classes': 'sequential', 'adm_in_channels': 2816, 'dtype': torch.float16, 'in_channels': 4, 'model_channels': 320,
                'num_res_blocks': [2, 2, 2], 'transformer_depth': [0, 0, 2, 2, 10, 10], 'channel_mult': [1, 2, 4], 'transformer_depth_middle': 10,
                'use_linear_in_transformer': True, 'context_dim': 2048, 'num_head_channels': 64, 'transformer_depth_output': [0, 0, 0, 2, 2, 2, 10, 10, 10],
                'use_temporal_attention': False, 'use_temporal_resblock': False}
    mapping = utils.unet_to_diffusers(SDXL)
    new_sd = copy.deepcopy(model_a)
    prefix = "model.diffusion_model."
    
    print("Merge begin")
    for k, v in mapping.items():
        # print(k, v)
        diffusion_model_key = f"{prefix}{v}"
        if diffusion_model_key not in model_b or k not in model_a:
            continue
        if model_b[diffusion_model_key].shape != model_a[k].shape:
            print("\nshape mismatch:")
            print(diffusion_model_key)
            print(model_b[diffusion_model_key].shape)
            print(k)
            print(model_a[k].shape)
            continue
        
        calc_weight = (1 - ratio) * model_a[k].to(torch.float32) + ratio * model_b[diffusion_model_key].to(torch.float32)
        new_sd[k] = calc_weight.to(torch.float16)
        if perturbed_ratio > 0:
            # perturbed model
            # code referenced from https://www.reddit.com/r/StableDiffusion/comments/1dfuicw/perturbed_sd3_experiment/
            new_sd[k] += torch.normal(torch.zeros_like(calc_weight)*calc_weight.mean(), torch.ones_like(calc_weight)*calc_weight.std()*perturbed_ratio).to(torch.float16)

    save_file(new_sd, merged_kolors_path, kolors_model.metadata())
    print("Merge End")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple Merging script")
    parser.add_argument(
        "--kolors_model_path",
        type=str,
        default=None,
        required=True,
        help="Kolors unet .safetensors file. Example: F:/models/Kolors/unet/diffusion_pytorch_model.fp16.safetensors",
    )
    parser.add_argument(
        "--sdxl_model_path",
        type=str,
        default=None,
        required=True,
        help="SDXL .safetensors file. Example: F:/models/Stable-diffusion/yoursdxlmodel.safetensors",
    )
    parser.add_argument(
        "--merged_kolors_path",
        type=str,
        default=None,
        required=True,
        help="Merged Kolors unet .safetensors file. Example: F:/models/Kolors/unet/merged_diffusion_pytorch_model.fp16.safetensors",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.2,
        help=("Merging Formula: Kolors weight * (1 - ratio) + SDXL weight * ratio"),
    )
    parser.add_argument(
        "--perturbed_ratio",
        type=float,
        default=0.02,
        help=("Experiment Function. Add some randomness to the merged. Default: 0. If you want to try it, recommanded 0.02 no more than 0.03"),
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args    

if __name__ == "__main__":
    args = parse_args()
    main(args)