# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import os
from typing import Callable, Dict, List, Optional, Union

import torch
from huggingface_hub.utils import validate_hf_hub_args

from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    get_submodule_by_name,
    is_bitsandbytes_available,
    is_gguf_available,
    is_peft_available,
    is_peft_version,
    is_torch_version,
    is_transformers_available,
    is_transformers_version,
    logging,
)
from diffusers.loaders.lora_base import (  # noqa
    LORA_WEIGHT_NAME,
    LORA_WEIGHT_NAME_SAFE,
    LoraBaseMixin,
    _fetch_state_dict,
    _load_lora_into_text_encoder,
    _pack_dict_with_prefix,
)
from diffusers.loaders.lora_conversion_utils import (
    _convert_bfl_flux_control_lora_to_diffusers,
    _convert_fal_kontext_lora_to_diffusers,
    _convert_hunyuan_video_lora_to_diffusers,
    _convert_kohya_flux_lora_to_diffusers,
    _convert_musubi_wan_lora_to_diffusers,
    _convert_non_diffusers_flux2_lora_to_diffusers,
    _convert_non_diffusers_hidream_lora_to_diffusers,
    _convert_non_diffusers_lora_to_diffusers,
    _convert_non_diffusers_ltxv_lora_to_diffusers,
    _convert_non_diffusers_lumina2_lora_to_diffusers,
    _convert_non_diffusers_qwen_lora_to_diffusers,
    _convert_non_diffusers_wan_lora_to_diffusers,
    _convert_xlabs_flux_lora_to_diffusers,
    _maybe_map_sgm_blocks_to_diffusers,
)

_LOW_CPU_MEM_USAGE_DEFAULT_LORA = False
if is_torch_version(">=", "1.9.0"):
    if (
        is_peft_available()
        and is_peft_version(">=", "0.13.1")
        and is_transformers_available()
        and is_transformers_version(">", "4.45.2")
    ):
        _LOW_CPU_MEM_USAGE_DEFAULT_LORA = True


logger = logging.get_logger(__name__)

TEXT_ENCODER_NAME = "text_encoder"
UNET_NAME = "unet"
TRANSFORMER_NAME = "transformer"

_MODULE_NAME_TO_ATTRIBUTE_MAP_FLUX = {"x_embedder": "in_channels"}

class ZImageLoraLoaderMixin(LoraBaseMixin):
    r"""
    Load LoRA layers into [`ZImageTransformer2DModel`]. Specific to [`ZImageLoraPipeline`].
    """

    _lora_loadable_modules = ["transformer"]
    transformer_name = TRANSFORMER_NAME

    @classmethod
    @validate_hf_hub_args
    def lora_state_dict(
        cls,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        **kwargs,
    ):
        r"""
        See [`~loaders.StableDiffusionLoraLoaderMixin.lora_state_dict`] for more details.
        """
        # Load the main state dict first which has the LoRA layers for either of
        # transformer and text encoder or both.
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        return_lora_metadata = kwargs.pop("return_lora_metadata", False)

        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True

        user_agent = {"file_type": "attn_procs_weights", "framework": "pytorch"}

        state_dict, metadata = _fetch_state_dict(
            pretrained_model_name_or_path_or_dict=pretrained_model_name_or_path_or_dict,
            weight_name=weight_name,
            use_safetensors=use_safetensors,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            token=token,
            revision=revision,
            subfolder=subfolder,
            user_agent=user_agent,
            allow_pickle=allow_pickle,
        )

        is_dora_scale_present = any("dora_scale" in k for k in state_dict)
        if is_dora_scale_present:
            warn_msg = "It seems like you are using a DoRA checkpoint that is not compatible in Diffusers at the moment. So, we are going to filter out the keys associated to 'dora_scale` from the state dict. If you think this is a mistake please open an issue https://github.com/huggingface/diffusers/issues/new."
            # logger.warning(warn_msg)
            print(warn_msg)
            state_dict = {k: v for k, v in state_dict.items() if "dora_scale" not in k}

        # has_alphas_in_sd = any(k.endswith(".alpha") for k in state_dict)
        # has_lora_unet = any(k.startswith("lora_unet_") for k in state_dict)
        # has_diffusion_model = any(k.startswith("diffusion_model.") for k in state_dict)
        # has_default = any("default." in k for k in state_dict)
        # if has_alphas_in_sd or has_lora_unet or has_diffusion_model or has_default:
        #     state_dict = _convert_non_diffusers_qwen_lora_to_diffusers(state_dict)

        out = (state_dict, metadata) if return_lora_metadata else state_dict
        return out

    # Copied from diffusers.loaders.lora_pipeline.CogVideoXLoraLoaderMixin.load_lora_weights
    def load_lora_weights(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        adapter_name: Optional[str] = None,
        hotswap: bool = False,
        **kwargs,
    ):
        """
        See [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] for more details.
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT_LORA)
        if low_cpu_mem_usage and is_peft_version("<", "0.13.0"):
            raise ValueError(
                "`low_cpu_mem_usage=True` is not compatible with this `peft` version. Please update it with `pip install -U peft`."
            )

        # if a dict is passed, copy it instead of modifying it inplace
        if isinstance(pretrained_model_name_or_path_or_dict, dict):
            pretrained_model_name_or_path_or_dict = pretrained_model_name_or_path_or_dict.copy()

        # First, ensure that the checkpoint is a compatible one and can be successfully loaded.
        kwargs["return_lora_metadata"] = True
        state_dict, metadata = self.lora_state_dict(pretrained_model_name_or_path_or_dict, **kwargs)

        is_correct_format = all("lora" in key for key in state_dict.keys())
        if not is_correct_format:
            raise ValueError("Invalid LoRA checkpoint.")

        self.load_lora_into_transformer(
            state_dict,
            transformer=getattr(self, self.transformer_name) if not hasattr(self, "transformer") else self.transformer,
            adapter_name=adapter_name,
            metadata=metadata,
            _pipeline=self,
            low_cpu_mem_usage=low_cpu_mem_usage,
            hotswap=hotswap,
        )

    @classmethod
    # Copied from diffusers.loaders.lora_pipeline.SD3LoraLoaderMixin.load_lora_into_transformer with SD3Transformer2DModel->QwenImageTransformer2DModel
    def load_lora_into_transformer(
        cls,
        state_dict,
        transformer,
        adapter_name=None,
        _pipeline=None,
        low_cpu_mem_usage=False,
        hotswap: bool = False,
        metadata=None,
    ):
        """
        See [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_into_unet`] for more details.
        """
        if low_cpu_mem_usage and is_peft_version("<", "0.13.0"):
            raise ValueError(
                "`low_cpu_mem_usage=True` is not compatible with this `peft` version. Please update it with `pip install -U peft`."
            )

        # Load the layers corresponding to transformer.
        logger.info(f"Loading {cls.transformer_name}.")
        transformer.load_lora_adapter(
            state_dict,
            network_alphas=None,
            prefix=TRANSFORMER_NAME,
            adapter_name=adapter_name,
            metadata=metadata,
            _pipeline=_pipeline,
            low_cpu_mem_usage=low_cpu_mem_usage,
            hotswap=hotswap,
        )

    @classmethod
    # Copied from diffusers.loaders.lora_pipeline.CogVideoXLoraLoaderMixin.save_lora_weights
    def save_lora_weights(
        cls,
        save_directory: Union[str, os.PathLike],
        transformer_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,
        is_main_process: bool = True,
        weight_name: str = None,
        save_function: Callable = None,
        safe_serialization: bool = True,
        transformer_lora_adapter_metadata: Optional[dict] = None,
    ):
        r"""
        See [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] for more information.
        """
        lora_layers = {}
        lora_metadata = {}

        if transformer_lora_layers:
            lora_layers[cls.transformer_name] = transformer_lora_layers
            lora_metadata[cls.transformer_name] = transformer_lora_adapter_metadata

        if not lora_layers:
            raise ValueError("You must pass at least one of `transformer_lora_layers` or `text_encoder_lora_layers`.")

        cls._save_lora_weights(
            save_directory=save_directory,
            lora_layers=lora_layers,
            lora_metadata=lora_metadata,
            is_main_process=is_main_process,
            weight_name=weight_name,
            save_function=save_function,
            safe_serialization=safe_serialization,
        )

    # Copied from diffusers.loaders.lora_pipeline.CogVideoXLoraLoaderMixin.fuse_lora
    def fuse_lora(
        self,
        components: List[str] = ["transformer"],
        lora_scale: float = 1.0,
        safe_fusing: bool = False,
        adapter_names: Optional[List[str]] = None,
        **kwargs,
    ):
        r"""
        See [`~loaders.StableDiffusionLoraLoaderMixin.fuse_lora`] for more details.
        """
        super().fuse_lora(
            components=components,
            lora_scale=lora_scale,
            safe_fusing=safe_fusing,
            adapter_names=adapter_names,
            **kwargs,
        )

    # Copied from diffusers.loaders.lora_pipeline.CogVideoXLoraLoaderMixin.unfuse_lora
    def unfuse_lora(self, components: List[str] = ["transformer"], **kwargs):
        r"""
        See [`~loaders.StableDiffusionLoraLoaderMixin.unfuse_lora`] for more details.
        """
        super().unfuse_lora(components=components, **kwargs)
