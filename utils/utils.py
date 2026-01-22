import os
import sys
import torch.nn as nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
import torch.nn.functional as F
import torch
import torch.distributed as dist
import re
import math
from collections.abc import Iterable
from itertools import repeat
from torchvision import transforms as T
from torchvision.transforms import functional as TF
import random
from PIL import Image
from hashlib import md5
import cv2
import numpy as np
from typing import List, Tuple, Union
from typing import List
import glob

def get_image_files(target_dir: str, image_extensions: list = None) -> list:
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif']
    
    # Build glob pattern: match all files with image extensions (case-insensitive via glob's `*`)
    patterns = [os.path.join(target_dir, f"*{ext}") for ext in image_extensions]
    # Add uppercase variants just in case (some OSes are case-sensitive)
    # patterns += [os.path.join(target_dir, f"*{ext.upper()}") for ext in image_extensions if ext.upper() != ext]
    
    # Use glob to get candidate files
    candidate_files = []
    for pattern in patterns:
        candidate_files.extend(glob.glob(pattern, recursive=False))
    
    return candidate_files

def find_image_files_by_regex(candidate_files: list, regex_pattern: str, image_extensions: list = None) -> list:
    """
    Find image files in `target_dir` that match `regex_pattern`.
    
    Args:
        target_dir (str): Directory to search.
        regex_pattern (str): Regex pattern to match full file *names* (not paths).
        image_extensions (list, optional): Allowed image extensions (case-insensitive).
                                           Defaults to common formats.
    
    Returns:
        list: Sorted list of absolute paths to matching image files.
    """

    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif']
    
    # Normalize extensions to lowercase for case-insensitive matching
    image_exts_lower = {ext.lower() for ext in image_extensions}
    
    # Compile regex for efficiency
    compiled_re = re.compile(regex_pattern)
    
    # Filter: keep only files whose *basename* matches regex AND has valid image extension
    matched_files = []
    for f in candidate_files:
        basename = os.path.basename(f)
        _, ext = os.path.splitext(basename)
        if ext.lower() in image_exts_lower and compiled_re.search(basename):
            matched_files.append(os.path.abspath(f))
    
    return sorted(matched_files)

def linear_interpolation(
    learning_target: torch.Tensor,      # (B, 16, F, H, W)
    reference_list: torch.Tensor,       # (B, 16, F, H, W)
    reasoning_frame: int,                # e.g., 6
    gamma: float = 0.5                  # >1 to bias toward target
) -> List[torch.Tensor]:
    assert reference_list.shape == learning_target.shape
    assert reasoning_frame >= 0
    B, C, F, H, W = learning_target.shape
    assert F == 1, "Only single-frame interpolation supported"

    ref = reference_list.squeeze(2)      # (B, C, H, W)
    tgt = learning_target.squeeze(2)     # (B, C, H, W)

    total_steps = reasoning_frame + 2
    # alphas = torch.linspace(0.0, 1.0, steps=total_steps, device=ref.device)
    t = torch.linspace(0.0, 1.0, steps=total_steps, device=ref.device)
    alphas = t ** gamma  # Non-linear spacing biased toward target

    latents = []
    for alpha in alphas:
        if alpha > 0.9:  # 纯参考帧
            # skip reference
            continue
        interp = (1 - alpha) * tgt + alpha * ref  # (B, C, H, W)
        interp = interp.unsqueeze(2)              # (B, C, 1, H, W)
        latents.append(interp)

    return latents

class ToTensorUniversal:
    """
    Convert PIL Image, NumPy array or torch tensor
    (uint8 or uint16, HWC or CHW) → float32 CHW tensor in [0, 1].
    """
    def __call__(self, pic):
        # --- PIL Image -------------------------------------------------------
        if isinstance(pic, Image.Image):
            return TF.to_tensor(pic)          # already returns CHW float32/255

        # --- NumPy array -----------------------------------------------------
        if isinstance(pic, np.ndarray):
            if pic.ndim == 3 and pic.shape[-1] in (3, 1):   # HWC
                pic = pic.transpose(2, 0, 1)                # → CHW
            if pic.dtype == np.uint8:
                pic = pic.astype(np.float32) / 255.0
            elif pic.dtype == np.uint16:
                pic = pic.astype(np.float32) / 65535.0
            else:
                pic = pic.astype(np.float32)
            return torch.from_numpy(pic)

        # --- torch tensor ----------------------------------------------------
        if isinstance(pic, torch.Tensor):
            if pic.ndim == 3 and pic.shape[-1] in (3, 1):   # HWC
                pic = pic.permute(2, 0, 1)                  # → CHW
            if pic.dtype == torch.uint8:
                pic = pic.float() / 255.0
            elif pic.dtype == torch.uint16:
                pic = pic.float() / 65535.0
            else:
                pic = pic.float()
            return pic

        raise TypeError(f"Unsupported input type: {type(pic)}")

# resize_method: str, "fs_resize" or "lanczos"
def resize(img: np.ndarray, resolution, resize_method="lanczos") -> np.ndarray:
    f_width, f_height = resolution
    if img.dtype != np.uint8: 
        img = cv2.convertScaleAbs(img)
        
    img_pil = Image.fromarray(img)
    resized_img = img_pil.resize((f_width, f_height), Image.LANCZOS)
    resized_img = np.array(resized_img)
    return resized_img # 返回最终（可能已校正）的图像

def find_index_from_right(lst, value):
    try:
        reversed_index = lst[::-1].index(value[::-1])
        return len(lst) - len(value) - reversed_index
    except:
        return -1
def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)

def set_grad_checkpoint(model, use_fp32_attention=False, gc_step=1):
    assert isinstance(model, nn.Module)

    def set_attr(module):
        module.grad_checkpointing = True
        module.fp32_attention = use_fp32_attention
        module.grad_checkpointing_step = gc_step
    model.apply(set_attr)


def auto_grad_checkpoint(module, *args, **kwargs):
    if getattr(module, 'grad_checkpointing', False):
        if isinstance(module, Iterable):
            gc_step = module[0].grad_checkpointing_step
            return checkpoint_sequential(module, gc_step, *args, **kwargs)
        else:
            return checkpoint(module, *args, **kwargs)
    return module(*args, **kwargs)


def checkpoint_sequential(functions, step, input, *args, **kwargs):

    # Hack for keyword-only parameter in a python 2.7-compliant way
    preserve = kwargs.pop('preserve_rng_state', True)
    if kwargs:
        raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))

    def run_function(start, end, functions):
        def forward(input):
            for j in range(start, end + 1):
                input = functions[j](input, *args)
            return input
        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = list(functions.children())

    # the last chunk has to be non-volatile
    end = -1
    segment = len(functions) // step
    for start in range(0, step * (segment - 1), step):
        end = start + step - 1
        input = checkpoint(run_function(start, end, functions), input, preserve_rng_state=preserve)
    return run_function(end + 1, len(functions) - 1, functions)(input)


def window_partition(x, window_size):
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    """
    Window unpartition into original sequences and removing padding.
    Args:
        x (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size, k_size, rel_pos):
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size):
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn

def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, tensor.ndim)))


#################################################################################
#                          Token Masking and Unmasking                          #
#################################################################################
def get_mask(batch, length, mask_ratio, device, mask_type=None, data_info=None, extra_len=0):
    """
    Get the binary mask for the input sequence.
    Args:
        - batch: batch size
        - length: sequence length
        - mask_ratio: ratio of tokens to mask
        - data_info: dictionary with info for reconstruction
    return:
        mask_dict with following keys:
        - mask: binary mask, 0 is keep, 1 is remove
        - ids_keep: indices of tokens to keep
        - ids_restore: indices to restore the original order
    """
    assert mask_type in ['random', 'fft', 'laplacian', 'group']
    mask = torch.ones([batch, length], device=device)
    len_keep = int(length * (1 - mask_ratio)) - extra_len

    if mask_type == 'random' or mask_type == 'group':
        noise = torch.rand(batch, length, device=device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        ids_removed = ids_shuffle[:, len_keep:]

    elif mask_type in ['fft', 'laplacian']:
        if 'strength' in data_info:
            strength = data_info['strength']

        else:
            N = data_info['N'][0]
            img = data_info['ori_img']
            # 获取原图的尺寸信息
            _, C, H, W = img.shape
            if mask_type == 'fft':
                # 对图片进行reshape，将其变为patch (3, H/N, N, W/N, N)
                reshaped_image = img.reshape((batch, -1, H // N, N, W // N, N))
                fft_image = torch.fft.fftn(reshaped_image, dim=(3, 5))
                # 取绝对值并求和获取频率强度
                strength = torch.sum(torch.abs(fft_image), dim=(1, 3, 5)).reshape((batch, -1,))
            elif type == 'laplacian':
                laplacian_kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32).reshape(1, 1, 3, 3)
                laplacian_kernel = laplacian_kernel.repeat(C, 1, 1, 1)
                # 对图片进行reshape，将其变为patch (3, H/N, N, W/N, N)
                reshaped_image = img.reshape(-1, C, H // N, N, W // N, N).permute(0, 2, 4, 1, 3, 5).reshape(-1, C, N, N)
                laplacian_response = F.conv2d(reshaped_image, laplacian_kernel, padding=1, groups=C)
                strength = laplacian_response.sum(dim=[1, 2, 3]).reshape((batch, -1,))

        # 对频率强度进行归一化，然后使用torch.multinomial进行采样
        probabilities = strength / (strength.max(dim=1)[0][:, None]+1e-5)
        ids_shuffle = torch.multinomial(probabilities.clip(1e-5, 1), length, replacement=False)
        ids_keep = ids_shuffle[:, :len_keep]
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_removed = ids_shuffle[:, len_keep:]

    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return {'mask': mask,
            'ids_keep': ids_keep,
            'ids_restore': ids_restore,
            'ids_removed': ids_removed}


def mask_out_token(x, ids_keep, ids_removed=None):
    """
    Mask out the tokens specified by ids_keep.
    Args:
        - x: input sequence, [N, L, D]
        - ids_keep: indices of tokens to keep
    return:
        - x_masked: masked sequence
    """
    N, L, D = x.shape  # batch, length, dim
    x_remain = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    if ids_removed is not None:
        x_masked = torch.gather(x, dim=1, index=ids_removed.unsqueeze(-1).repeat(1, 1, D))
        return x_remain, x_masked
    else:
        return x_remain


def mask_tokens(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


def unmask_tokens(x, ids_restore, mask_token):
    # x: [N, T, D] if extras == 0 (i.e., no cls token) else x: [N, T+1, D]
    mask_tokens = mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
    x = torch.cat([x, mask_tokens], dim=1)
    x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
    return x


# Parse 'None' to None and others to float value
def parse_float_none(s):
    assert isinstance(s, str)
    return None if s == 'None' else float(s)


#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges


def init_processes(fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = str(random.randint(2000, 6000))
    print(f'MASTER_ADDR = {os.environ["MASTER_ADDR"]}')
    print(f'MASTER_PORT = {os.environ["MASTER_PORT"]}')
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', rank=args.global_rank, world_size=args.global_size)
    fn(args)
    if args.global_size > 1:
        cleanup()


def mprint(*args, **kwargs):
    """
    Print only from rank 0.
    """
    if dist.get_rank() == 0:
        print(*args, **kwargs)


def cleanup():
    """
    End DDP training.
    """
    dist.barrier()
    mprint("Done!")
    dist.barrier()
    dist.destroy_process_group()


#----------------------------------------------------------------------------
# logging info.
class Logger(object):
    """
    Redirect stderr to stdout, optionally print stdout to a file,
    and optionally force flushing on both stdout and the file.
    """

    def __init__(self, file_name=None, file_mode="w", should_flush=True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def write(self, text):
        """Write text to stdout (and a file) and optionally flush."""
        if len(text) == 0: # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self):
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self):
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()


class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])


def prepare_prompt_ar(prompt, ratios, device='cpu', show=True):
    # get aspect_ratio or ar
    aspect_ratios = re.findall(r"--aspect_ratio\s+(\d+:\d+)", prompt)
    ars = re.findall(r"--ar\s+(\d+:\d+)", prompt)
    custom_hw = re.findall(r"--hw\s+(\d+:\d+)", prompt)
    if show:
        print("aspect_ratios:", aspect_ratios, "ars:", ars, "hws:", custom_hw)
    prompt_clean = prompt.split("--aspect_ratio")[0].split("--ar")[0].split("--hw")[0]
    if len(aspect_ratios) + len(ars) + len(custom_hw) == 0 and show:
        print("Wrong prompt format. Set to default ar: 1. change your prompt into format '--ar h:w or --hw h:w' for correct generating")
    if len(aspect_ratios) != 0:
        ar = float(aspect_ratios[0].split(':')[0]) / float(aspect_ratios[0].split(':')[1])
    elif len(ars) != 0:
        ar = float(ars[0].split(':')[0]) / float(ars[0].split(':')[1])
    else:
        ar = 1.
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - ar))
    if len(custom_hw) != 0:
        custom_hw = [float(custom_hw[0].split(':')[0]), float(custom_hw[0].split(':')[1])]
    else:
        custom_hw = ratios[closest_ratio]
    default_hw = ratios[closest_ratio]
    prompt_show = f'prompt: {prompt_clean.strip()}\nSize: --ar {closest_ratio}, --bin hw {ratios[closest_ratio]}, --custom hw {custom_hw}'
    return prompt_clean, prompt_show, torch.tensor(default_hw, device=device)[None], torch.tensor([float(closest_ratio)], device=device)[None], torch.tensor(custom_hw, device=device)[None]


def resize_and_crop_tensor(samples: torch.Tensor, new_width: int, new_height: int):
    orig_hw = torch.tensor([samples.shape[2], samples.shape[3]], dtype=torch.int)
    custom_hw = torch.tensor([int(new_height), int(new_width)], dtype=torch.int)

    if (orig_hw != custom_hw).all():
        ratio = max(custom_hw[0] / orig_hw[0], custom_hw[1] / orig_hw[1])
        resized_width = int(orig_hw[1] * ratio)
        resized_height = int(orig_hw[0] * ratio)

        transform = T.Compose([
            T.Resize((resized_height, resized_width)),
            T.CenterCrop(custom_hw.tolist())
        ])
        return transform(samples)
    else:
        return samples


def resize_and_crop_img(img: Image, new_width, new_height):
    orig_width, orig_height = img.size

    ratio = max(new_width/orig_width, new_height/orig_height)
    resized_width = int(orig_width * ratio)
    resized_height = int(orig_height * ratio)

    img = img.resize((resized_width, resized_height), Image.LANCZOS)

    left = (resized_width - new_width)/2
    top = (resized_height - new_height)/2
    right = (resized_width + new_width)/2
    bottom = (resized_height + new_height)/2

    img = img.crop((left, top, right, bottom))

    return img



def mask_feature(emb, mask):
    if emb.shape[0] == 1:
        keep_index = mask.sum().item()
        return emb[:, :, :keep_index, :], keep_index
    else:
        masked_feature = emb * mask[:, None, :, None]
        return masked_feature, emb.shape[2]
    

def replace_non_utf8_characters(filepath):
    content = ""
    # Helper function to filter out non-UTF-8 characters
    # def clean_text(text):
    #     return ''.join([char if ord(char) < 128 else '' for char in text])

    try:
        # Read file content
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Clean content by removing non-UTF-8 characters
        cleaned_content = content.decode('utf-8')

        # Write cleaned content back to the file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
    
    return cleaned_content

def get_md5_by_path(file_path):
    try:
        with open(file_path, 'rb') as f:
            return md5(f.read()).hexdigest()
    except:
        print(f"Error getting md5 for {file_path}")
        return ''
    

def sample_reference_timesteps(
    t: float,
    t_low: float = 0.15,
    t_high: float = 0.85,
    max_references: int = 3,
    min_step: float = 0.05,
    max_step: float = 0.1,
    seed: int = None
) -> list[float]:
    """
    根据当前时间步 t 动态采样参考时间步。
    
    Args:
        t: 当前归一化时间步 (0=清晰, 1=噪声)
        t_low: 有效区间的下界（低于此值无参考）
        t_high: 有效区间的上界（高于此值无参考）
        max_references: 区间中心最多参考步数
        min_step: 最小时间步间隔
        max_step: 最大时间步间隔
        seed: 随机种子（可选）
    
    Returns:
        list[float]: 参考时间步列表（按时间递增排序），若无参考则返回 []
    """
    if seed is not None:
        random.seed(seed)
    
    # 1. 区间左侧外：无参考
    if t < t_low:
        return []
    
    # 2. 区间右侧外：返回 1 - t
    if t >= (t_high - min_step):
        return [ 0.0001 ]
    
    # 2. 计算当前位置在有效区间中的归一化位置 (0=左边界, 1=右边界)
    interval_length = t_high - t_low
    interval_pos = (t - t_low) / interval_length  # ∈ (0, 1)
    
    # 3. 距离最近边界的归一化距离 ∈ (0, 0.5]
    distance_to_edge = min(interval_pos, 1 - interval_pos)
    
    # 4. 线性映射到参考数量：0.0 → 0, 0.5 → max_references
    ref_count_float = max_references * (2 * distance_to_edge)  # ∈ (0, max_references]
    ref_count = max(1, min(max_references, int(round(ref_count_float))))
    
    # 5. 采样参考时间步（必须 > t，且 <= t_high）
    references = []
    current_t = t
    for _ in range(ref_count):
        available_range = t_high - current_t  # 关键修正：上限是 t_high，不是 1.0
        if available_range < min_step:
            break  # 无法再放置一个有效参考
        
        step_upper = min(max_step, available_range)
        if step_upper <= min_step:
            step = min_step
        else:
            step = random.uniform(min_step, step_upper)
        
        ref_t = current_t + step
        # 理论上 ref_t <= t_high，但加个保险
        if ref_t > t_high:
            break
        
        references.append(ref_t)
        current_t = ref_t
    
    return sorted(references)


def simple_center_crop(image,scale_with_height,closest_resolution):
    height, width, _ = image.shape
    # print("ori size:",width,height)
    if scale_with_height: 
        up_scale = height / closest_resolution[1]
    else:
        up_scale = width / closest_resolution[0]

    expanded_closest_size = (int(closest_resolution[0] * up_scale + 0.5), int(closest_resolution[1] * up_scale + 0.5))
    
    diff_x = abs(expanded_closest_size[0] - width)
    diff_y = abs(expanded_closest_size[1] - height)

    crop_x = 0
    crop_y = 0
    # crop extra part of the resized images
    if diff_x>0:
        crop_x =  diff_x //2
        cropped_image = image[:,  crop_x:width-diff_x+crop_x]
    elif diff_y>0:
        crop_y =  diff_y//2
        cropped_image = image[crop_y:height-diff_y+crop_y, :]
    else:
        # 1:1 ratio
        cropped_image = image

    # print(f"ori ratio:{width/height}")
    height, width, _ = cropped_image.shape  
    return resize(cropped_image, closest_resolution, resize_method="lanczos"), crop_x, crop_y


def read_image(image_path):
    try:
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is not None:
            # Convert to RGB format (assuming the original image is in BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            print(f"Failed to open {image_path}.")
    except Exception as e:
        print(f"An error occurred while processing {image_path}: {e}")
    return image


# return closest_ratio and width,height closest_resolution
def get_nearest_resolution_utils(image, resolution_set):
    height, width, _ = image.shape
    # get ratio
    image_ratio = width / height

    target_set = resolution_set.copy()
    reversed_set = [(y, x) for x, y in target_set]
    target_set = sorted(set(target_set + reversed_set))
    target_ratio = sorted(set([round(width/height, 2) for width,height in target_set]))
    
    # Find the closest vertical ratio
    closest_ratio = min(target_ratio, key=lambda x: abs(x - image_ratio))
    closest_resolution = target_set[target_ratio.index(closest_ratio)]

    return closest_ratio,closest_resolution

def crop_image_utils(image_path,resolution_set):
    image = read_image(image_path)
    ##############################################################################
    # Simple center crop for others
    ##############################################################################
    # width, height = image.size
    # original_size = (height, width)
    # image = numpy.array(image)
    
    height, width, _ = image.shape
    # original_size = (height, width)
    
    # get nearest resolution
    closest_ratio,closest_resolution = get_nearest_resolution_utils(image,resolution_set)
    # we need to expand the closest resolution to target resolution before cropping
    scale_ratio = closest_resolution[0] / closest_resolution[1]
    image_ratio = width / height
    scale_with_height = True
    # crops_coords_top_left = (0,0)
    # referenced kohya ss code
    if image_ratio < scale_ratio: 
        scale_with_height = False
    try:
        # image = simple_center_crop(image,scale_with_height,closest_resolution)
        image,crop_x,crop_y = simple_center_crop(image,scale_with_height,closest_resolution)
        # crops_coords_top_left = (crop_y,crop_x)
        # save_webp(simple_crop_image,filename,'simple',os.path.join(output_dir,"simple"))
    except Exception as e:
        print(e)
        raise e
    # test = Image.fromarray(image)
    # test.show()
    # set meta data
    return image



@torch.no_grad()
def vae_encode_utils(vae,image, vae_type="wan"):
    # create tensor latent
    
    pixel_values = []
    pixel_values.append(image)
    pixel_values = torch.stack(pixel_values).to(vae.device)
    # del image
    
    if vae_type=="wan":
        # Qwen expects a `num_frames` dimension too.
        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(2)
    
    latent = vae.encode(pixel_values).latent_dist.sample().squeeze(0)
    
    del pixel_values
    latent_dict = {
        'latent': latent.cpu()
    }
    return latent_dict

def project_timestep_snr(
    t_actual: torch.Tensor,
    comfort_zone_steps: Union[List[float], torch.Tensor],
    schedule: str = "linear"
) -> torch.Tensor:
    """
    修正版SNR投影函数 - 张量优先设计
    1. 支持浮点输入 (874.9999)
    2. 精确处理0步语义
    3. 自动四舍五入到最近舒适步
    """
    if not isinstance(t_actual, torch.Tensor):
        raise TypeError("t_actual must be a torch.Tensor")
    if t_actual.dim() not in [1, 2]:
        raise ValueError(f"t_actual must be 1D or 2D tensor, got shape {t_actual.shape}")
    
    # 保存原始形状，展平处理
    original_shape = t_actual.shape
    t_flat = t_actual.reshape(-1)
    
    device = t_flat.device
    
    # 1. 预处理舒适区：转换为张量 + 四舍五入 + 去重 + 排序
    if not isinstance(comfort_zone_steps, torch.Tensor):
        comfort_zone_steps = torch.tensor(comfort_zone_steps, device=device)
    
    # 四舍五入到最近整数，钳位到[0,1000]
    comfort_steps = torch.round(comfort_zone_steps).long().clamp(0, 1000)
    comfort_steps = torch.unique(comfort_steps.sort().values)
    
    if len(comfort_steps) == 0:
        raise ValueError("No valid comfort steps after filtering. Must be between 0-1000")
    
    # 2. 处理输入时间步：四舍五入 + 钳位
    t_rounded = torch.round(t_flat).long().clamp(0, 1000)
    
    # 3. 核心投影逻辑
    # 3.1 特殊处理0步 (关键!)
    has_zero = (comfort_steps == 0).any()
    comfort_nonzero = comfort_steps[comfort_steps > 0]
    
    # 3.2 初始化投影结果
    t_proj = torch.empty_like(t_rounded)
    
    if has_zero and len(comfort_nonzero) > 0:
        last_nonzero = comfort_nonzero.min()  # 最小的非零舒适步 (300)
        threshold = last_nonzero // 2         # 150 for SDXL
        
        # 规则1: t=0 必须映射到0
        zero_mask = (t_rounded == 0)
        t_proj[zero_mask] = 0
        
        # 规则2: t <= threshold 且 t>0 -> 用SNR决策
        candidate_mask = (t_rounded > 0) & (t_rounded <= threshold)
        if candidate_mask.any():
            # 获取SNR缓存
            T = 1000
            cache_key = (schedule, str(device))
            
            if not hasattr(project_timestep_snr, "snr_cache"):
                project_timestep_snr.snr_cache = {}
            
            if cache_key not in project_timestep_snr.snr_cache:
                if schedule == "linear":
                    beta_start, beta_end = 0.00085, 0.012
                    betas = torch.linspace(beta_start**0.5, beta_end**0.5, T, device=device)**2
                    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
                elif schedule == "cosine":
                    s = 0.008
                    x = torch.linspace(0, T, T + 1, device=device)
                    f_t = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
                    alphas_cumprod = f_t / f_t[0]
                    alphas_cumprod = torch.clamp(alphas_cumprod[1:], 0.0001, 0.9999)
                elif schedule == "scaled_linear":
                    beta_start, beta_end = 0.0001, 0.02
                    betas = torch.linspace(beta_start**0.5, beta_end**0.5, T, device=device)**2
                    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
                else:
                    raise ValueError(f"Unsupported schedule: {schedule}")
                
                sigmas = torch.sqrt(1.0 - alphas_cumprod)
                snrs = torch.log(alphas_cumprod / (sigmas ** 2 + 1e-8))
                project_timestep_snr.snr_cache[cache_key] = snrs
            
            snrs = project_timestep_snr.snr_cache[cache_key]
            
            # 获取候选步的SNR
            candidate_ts = t_rounded[candidate_mask]
            candidate_snrs = snrs[candidate_ts - 1]  # t=1使用snrs[0]
            
            # 0步的SNR (理论+∞)
            zero_snr = 1e6
            
            # 300步的SNR
            nonzero_snr = snrs[last_nonzero - 1]
            
            # 决策：如果候选步的SNR更接近0步或SNR>5.0则映射到0
            dist_to_zero = torch.abs(candidate_snrs - zero_snr)
            dist_to_nonzero = torch.abs(candidate_snrs - nonzero_snr)
            map_to_zero = (dist_to_zero < dist_to_nonzero) | (candidate_snrs > 5.0)
            
            # 应用映射
            t_proj_candidate = torch.where(
                map_to_zero,
                torch.zeros_like(candidate_ts),
                torch.full_like(candidate_ts, last_nonzero)
            )
            t_proj[candidate_mask] = t_proj_candidate
        
        # 规则3: 剩余步骤 (t > threshold) 用线性距离
        remaining_mask = ~(zero_mask | candidate_mask)
    else:
        remaining_mask = torch.ones_like(t_rounded, dtype=torch.bool)
    
    # 3.3 剩余步骤：使用线性距离
    if remaining_mask.any():
        remaining_ts = t_rounded[remaining_mask]
        dist_matrix = torch.abs(remaining_ts.unsqueeze(1) - comfort_steps.unsqueeze(0))
        nearest_idx = torch.argmin(dist_matrix, dim=1)
        t_proj[remaining_mask] = comfort_steps[nearest_idx]
    
    # 恢复原始形状
    return t_proj.reshape(original_shape).detach()

def parse_indices(indices_str):
    indices = []
    if indices_str.strip() == "":
        return []
    parts = indices_str.split(",")
    for part in parts:
        part = part.strip()
        
        # skip empty parts
        if not part:
            continue
        
        if "-" in part:
            # Handle range notation like "50-53"
            try:
                start, end = part.split("-")
                start_idx = int(start.strip())
                end_idx = int(end.strip())
                if start_idx > end_idx:
                    raise ValueError(f"Invalid range: {part} (start index greater than end index)")
                indices.extend(range(start_idx, end_idx + 1))
            except ValueError:
                raise ValueError(f"Invalid range format: {part}. Use format like '10-15'.")
        else:
            # Handle single integer
            try:
                indices.append(int(part))
            except ValueError:
                raise ValueError(f"Invalid layer index: {part}")
    return indices

def print_end_signal():
    print("=========== End Training ===========")

# Example usage:
if __name__ == "__main__":
    target_dir = r"F:\ImageSet\general_editing\style_gen_test\aelion"
    # Example: match files like "cat_001.jpg", "dog_2025.png", but NOT "cat_old.jpg"
    pattern = r".*_T\.(?:jpg|jpeg|png|bmp|tiff|tif|webp|gif)$"  # lowercase word + underscore + digits + .jpg/.png
    
    matches = find_image_files_by_regex(target_dir, pattern)
    for f in matches:
        print(f)
        