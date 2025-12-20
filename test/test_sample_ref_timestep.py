import random
import math


def sample_reference_timesteps(
    t: float,
    t_low: float = 0.15,
    t_high: float = 0.85,
    max_references: int = 3,
    min_step: float = 0.05,
    max_step: float = 0.2,
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
    
    # 1. 区间外：无参考
    # 1. 区间左侧外：无参考
    if t <= t_low:
        return []
    
    # 2. 区间右侧外：返回 [1 - t]
    if t >= t_high:
        # use clear timestep for high noise reference
        return [0.0001]
    
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


def main():
    """Test function for sample_reference_timesteps"""
    print("Testing sample_reference_timesteps function...\n")
    
    # Test 1: Basic functionality
    print("Test 1: Basic functionality")
    t = 0.999  # Middle of the valid range
    result = sample_reference_timesteps(t)
    result = result[0]
    timestep = int(1000 - result * 1000)
    print(f"Input: t={t}")
    print(f"Output: {result}\n")
    
    print(f"Input: t={timestep}")
    print(f"Output: {timestep}\n")
    # Test 2: Boundary conditions
    # print("Test 2: Boundary conditions")
    # for t in [0.0, 0.2, 0.25, 0.75, 0.8, 1.0]:
    #     result = sample_reference_timesteps(t)
    #     print(f"Input: t={t}, Output: {result}")
    # print()
    
    # # Test 3: Different seeds for reproducibility
    # print("Test 3: Testing with seed for reproducibility")
    # result1 = sample_reference_timesteps(0.5, seed=42)
    # result2 = sample_reference_timesteps(0.5, seed=42)
    # print(f"With seed=42: {result1}")
    # print(f"With seed=42 again: {result2}")
    # print(f"Results match: {result1 == result2}\n")
    
    # # Test 4: Different parameter values
    # print("Test 4: Different parameter values")
    # result = sample_reference_timesteps(0.5, max_references=5, min_step=0.1, max_step=0.3)
    # print(f"With custom parameters: {result}")
    # result = sample_reference_timesteps(0.5, t_low=0.1, t_high=0.9)
    # print(f"With custom range [0.1, 0.9]: {result}\n")
    
    # # Test 5: Edge cases and boundary conditions
    # print("Test 5: Edge cases and boundary conditions")
    
    # # Exactly at boundaries
    # print(f"At t_low (0.25): {sample_reference_timesteps(0.25)}")
    # print(f"At t_high (0.75): {sample_reference_timesteps(0.75)}")
    
    # # Close to boundaries but inside
    # print(f"Just above t_low (0.251): {sample_reference_timesteps(0.251)}")
    # print(f"Just below t_high (0.749): {sample_reference_timesteps(0.749)}")
    
    # # Extreme values
    # print(f"Very close to 0: {sample_reference_timesteps(0.001)}")
    # print(f"Very close to 1: {sample_reference_timesteps(0.999)}")
    # print()
    
    # # Test 6: Verify reference timesteps are always greater than current t
    # print("Test 6: Verify reference timesteps are always greater than current t")
    # for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
    #     refs = sample_reference_timesteps(t, seed=123)
    #     valid = all(ref > t for ref in refs)
    #     print(f"t={t}, references={refs}, all greater than t: {valid}")
    # print()
    
    # # Test 7: Different random seeds and parameter combinations
    # print("Test 7: Different random seeds and parameter combinations")
    
    # # Test with different seeds to see variation
    # print("Testing randomness with different seeds:")
    # base_t = 0.5
    # for seed in [1, 42, 123, 999]:
    #     result = sample_reference_timesteps(base_t, seed=seed)
    #     print(f"Seed {seed}: {result}")
    
    # print("\nTesting with various parameter combinations:")
    # test_params = [
    #     {"max_references": 1, "min_step": 0.05, "max_step": 0.1},
    #     {"max_references": 5, "min_step": 0.1, "max_step": 0.3},
    #     {"max_references": 2, "min_step": 0.02, "max_step": 0.05},
    #     {"t_low": 0.1, "t_high": 0.9, "max_references": 4}
    # ]
    
    # for i, params in enumerate(test_params):
    #     result = sample_reference_timesteps(0.5, seed=42, **params)
    #     print(f"Params {i+1} {params}: {result}")


if __name__ == "__main__":
    main()
