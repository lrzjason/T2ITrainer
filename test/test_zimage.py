import torch
import math
from typing import List, Tuple, Union
from flux.flux_utils import compute_density_for_timestep_sampling
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
)
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


def project_timestep_sequences_tensor(
    timestep_tensor: torch.Tensor,
    comfort_zone_steps: Union[List[float], torch.Tensor],
    schedule: str = "linear"
) -> torch.Tensor:
    """
    张量优先的序列投影函数
    输入:
        timestep_tensor: 1D或2D张量 (batch_size,) or (batch_size, num_steps)
        comfort_zone_steps: 舒适区步骤 (列表或张量)
    输出:
        同形状的投影后张量
    """
    return project_timestep_snr(timestep_tensor, comfort_zone_steps, schedule)



# ====================== 验证测试 ======================
if __name__ == "__main__":
    # 您提供的SDXL 9-step蒸馏步骤
    COMFORT_STEPS = [1000.0000,  954.5454,  900.0000,  833.3333,  750.0000,  642.8571,
                     500.0000,  300.0000,    0.0000]
    
    pretrained_model_name_or_path = r"F:\T2ITrainer_pulic\T2ITrainer\z_img_models\Z-Image-Turbo"
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path, subfolder="scheduler"
    )
    # batch_size = batch["batch_size"]
    u = compute_density_for_timestep_sampling(
        weighting_scheme="logit_normal",
        batch_size=1,
        logit_mean=0,
        logit_std=1,
        mode_scale=1.29
    )
    device = torch.device("cuda")
    indices = (u * noise_scheduler.config.num_train_timesteps).long()
    timesteps = noise_scheduler.timesteps[indices].to(device=device)
    
    print(timesteps)
    
    
    # 直接使用张量投影
    projected = project_timestep_sequences_tensor(
        timestep_tensor=timesteps,
        comfort_zone_steps=COMFORT_STEPS,
        schedule="linear"
    )
    print(f"投影结果 (1D): {projected}\n")
    
    