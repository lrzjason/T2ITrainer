# Thanks to Kwai/kolors for the open source weight and comfyanonymous/ComfyUI for unet mapping code

import torch
import safetensors
from safetensors import safe_open
from safetensors.torch import save_file
import json
import comfy.utils as utils
import copy
import argparse
import os

def main(args):
    print("running merge_state_dict")
    kolors_model_path = args.kolors_model_path
    convert_target_path = args.convert_target_path
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    print("kolors_model_path:",kolors_model_path)
    print("convert_target_path:",convert_target_path)
    kolors_model = safetensors.safe_open(kolors_model_path, 'pt')
    ori_keys = kolors_model.keys()
    model_ori = {key:kolors_model.get_tensor(key) for key in ori_keys}


    # convert_model = safetensors.safe_open(convert_target_path, 'pt')
    # keys = convert_model.keys()
    # # print(keys)
    # model_a = {key:convert_model.get_tensor(key) for key in keys}
    # model_b = safetensors.torch.load_file(sdxl_model_path)
    # from ComfyUI-Kolors-MZ plugin
    Kolors = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
            'num_classes': 'sequential', 'adm_in_channels': 5632, 'dtype': torch.float16, 'in_channels': 4, 'model_channels': 320,
            'num_res_blocks': [2, 2, 2], 'transformer_depth': [0, 0, 2, 2, 10, 10], 'channel_mult': [1, 2, 4], 'transformer_depth_middle': 10,
            'use_linear_in_transformer': True, 'context_dim': 2048, 'num_head_channels': 64, 'transformer_depth_output': [0, 0, 0, 2, 2, 2, 10, 10, 10],
            'use_temporal_attention': False, 'use_temporal_resblock': False}

    mapping = utils.unet_to_diffusers(Kolors)
    # new_sd = copy.deepcopy(model_a)
    # new_diffusers_weight = copy.deepcopy(model_ori)
    prefix = "model.diffusion_model."
    
    print("convert begin")
    missing_dict = {
        "encoder_hid_proj.bias": model_ori["encoder_hid_proj.bias"],
        "encoder_hid_proj.weight": model_ori["encoder_hid_proj.weight"]
    }
    err_k = ""
    err_v = ""
    # for k, v in mapping.items():
    #     if k not in ori_keys:
    #         print(k)
            # continue
    #     try:
    #         err_k = k
    #         err_v = v
    #         diffusion_model_key = f"{prefix}{v}"
    #         model_value = model_a[diffusion_model_key]
    #         new_diffusers_weight[k] = model_value
    #     except:
    #         print("convert error")
    #         print(err_k,err_v)
    
    save_file(missing_dict, f"F:/Comfyui-Kolors-Utils/missing_tensors.safetensors", kolors_model.metadata())
    print("convert End")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple convert script")
    parser.add_argument(
        "--kolors_model_path",
        type=str,
        default=None,
        required=False,
        help="Kolors unet .safetensors file. Example: F:/models/Kolors/unet/diffusion_pytorch_model.fp16.safetensors",
    )
    parser.add_argument(
        "--convert_target_path",
        type=str,
        default=None,
        required=False,
        help="Convert target path",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        required=False,
        help="save converted weight path",
    )
    # parser.add_argument(
    #     "--sdxl_model_path",
    #     type=str,
    #     default=None,
    #     required=True,
    #     help="SDXL .safetensors file. Example: F:/models/Stable-diffusion/yoursdxlmodel.safetensors",
    # )
    # parser.add_argument(
    #     "--merged_kolors_path",
    #     type=str,
    #     default=None,
    #     required=True,
    #     help="Merged Kolors unet .safetensors file. Example: F:/models/Kolors/unet/merged_diffusion_pytorch_model.fp16.safetensors",
    # )
    # parser.add_argument(
    #     "--ratio",
    #     type=float,
    #     default=0.2,
    #     help=("Merging Formula: Kolors weight * (1 - ratio) + SDXL weight * ratio"),
    # )
    # parser.add_argument(
    #     "--perturbed_ratio",
    #     type=float,
    #     default=0.02,
    #     help=("Experiment Function. Add some randomness to the merged. Default: 0. If you want to try it, recommanded 0.02 no more than 0.03"),
    # )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args    

if __name__ == "__main__":
    args = parse_args()
    args.kolors_model_path = "F:/models/unet/new_kolors/diffusion_pytorch_model.fp16.safetensors"
    args.convert_target_path = "F:/models/Stable-diffusion/sdxl/comfy_output_checkpoint/NijiKolorsAlphav01.safetensors"
    args.save_path = "F:/models/unet/NijiKolors/"
    main(args)