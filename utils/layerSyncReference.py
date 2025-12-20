import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
from peft import PeftModel

# ———————— 1. 自动探测总 block 数 & 层名 ————————
def get_total_blocks_and_names(model: nn.Module, block_pattern: str = "transformer_blocks") -> (int, List[str]):
    """
    自动探测模型中符合 block_pattern 的模块数量及完整命名路径。
    适用于: DiT/SiT/MMDiT/U-Net 等，支持任意嵌套结构。
    默认 pattern='transformer_blocks' 适用于大多数 ViT-based diffusion models.
    """
    block_names = []
    for name, module in model.named_modules():
        if block_pattern in name and hasattr(module, "forward") and not any(kw in name for kw in ["mlp", "norm", "proj"]):
            # 避免误抓子模块（如 layer_norm 在 block 内部）；只抓顶层 block
            if "." in name:
                parent = name.rsplit(".", 1)[0]
                if parent in block_names:
                    continue  # skip submodules
            block_names.append(name)
    block_names = sorted(block_names, key=lambda x: int(x.split(".")[-1]) if x.split(".")[-1].isdigit() else float('inf'))
    total_blocks = len(block_names)
    return total_blocks, block_names

# ———————— 2. LayerSync 层对选择策略 ————————
def get_layer_sync_pair(
    total_blocks: int,
    alpha: float = 0.4,
    beta: float = 0.7,
    min_gap: int = 10,
    exclude_last_ratio: float = 0.2
) -> (int, int):
    """
    返回 weak_layer_idx, strong_layer_idx （均为 0-based index）
    """
    assert total_blocks > 0, "Model has no blocks!"
    assert 0 < alpha < beta < 1 - exclude_last_ratio, f"beta={beta} must be < {1 - exclude_last_ratio}"
    
    # 强层上限（避开最后 exclude_last_ratio * total）
    max_strong_idx = int((1 - exclude_last_ratio) * total_blocks) - 1
    
    weak = int(alpha * total_blocks)
    strong = min(int(beta * total_blocks), max_strong_idx)
    
    # 保证最小间隔
    if strong - weak < min_gap:
        strong = weak + min_gap
    
    # 边界检查
    weak = max(0, min(weak, total_blocks - 2))
    strong = max(weak + min_gap, min(strong, total_blocks - 1))
    
    return weak, strong

# ———————— 3. Hook 提取器（支持 PeftModel） ————————
class LayerExtractor:
    def __init__(self):
        self.activations: Dict[str, torch.Tensor] = {}

    def register_hooks(self, model: nn.Module, layer_names: List[str]):
        """
        注册 forward hook，支持 PeftModel（hook 仍作用于 base_model）
        """
        def make_hook(name):
            def hook(_module, _input, output):
                # output: (B, N, D) or tuple — 取第一个主输出
                out = output[0] if isinstance(output, (tuple, list)) else output
                self.activations[name] = out
            return hook

        hooked = 0
        for name, module in model.named_modules():
            if name in layer_names:
                module.register_forward_hook(make_hook(name))
                hooked += 1
        if hooked == 0:
            raise ValueError(f"No layers matched names: {layer_names}")
        print(f"[LayerExtractor] Registered hooks on {hooked} layers: {layer_names}")

    def get_activation(self, name: str) -> torch.Tensor:
        if name not in self.activations:
            raise KeyError(f"Activation for '{name}' not found. Forward not called or hook failed.")
        return self.activations[name]

    def clear(self):
        self.activations.clear()

# ———————— 4. 一键初始化 LayerSync Hook ————————
def init_layer_sync_extractor(
    model: nn.Module,
    block_pattern: str = "transformer_blocks",
    alpha: float = 0.4,
    beta: float = 0.7,
    min_gap: int = 10
) -> (LayerExtractor, str, str):
    """
    返回：extractor, weak_layer_name, strong_layer_name
    """
    total_blocks, block_names = get_total_blocks_and_names(model, block_pattern=block_pattern)
    print(f"[LayerSync] Detected {total_blocks} blocks with pattern '{block_pattern}'")

    if total_blocks == 0:
        raise RuntimeError("No blocks detected. Check `block_pattern`.")

    weak_idx, strong_idx = get_layer_sync_pair(
        total_blocks=total_blocks,
        alpha=alpha,
        beta=beta,
        min_gap=min_gap
    )
    weak_name = block_names[weak_idx]
    strong_name = block_names[strong_idx]

    extractor = LayerExtractor()
    extractor.register_hooks(model, [weak_name, strong_name])

    print(f"[LayerSync] Selected → weak: '{weak_name}' (idx {weak_idx}), strong: '{strong_name}' (idx {strong_idx})")
    return extractor, weak_name, strong_name

# ———————— 示例用法 ————————
if __name__ == "__main__":
    # 假设 model 是已加载的 PeftModel（或普通 nn.Module）
    # model = get_peft_model(Transformer2DModel(...), lora_config)

    # 自动初始化 LayerSync hook
    extractor, weak_layer, strong_layer = init_layer_sync_extractor(
        model,
        block_pattern="transformer_blocks",  # 对 SD3/MMDiT 有效；U-Net 可用 "down_blocks" / "up_blocks"
        alpha=0.4,
        beta=0.7,
        min_gap=12
    )

    # 在 train_step 中使用：
    # extractor.clear()
    # v_pred = model(xt, t, ...).sample
    # z_weak = extractor.get_activation(weak_layer)   # (B, P, D)
    # z_strong = extractor.get_activation(strong_layer) # (B, P, D)
    # loss_sync = -F.cosine_similarity(F.normalize(z_weak, dim=-1),
    #                                   F.normalize(z_strong.detach(), dim=-1), dim=-1).mean()
