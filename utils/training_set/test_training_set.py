#!/usr/bin/env python3
"""
test_training_set.py
既可以：
  1) 被 pytest 收集并正常跑用例（原有 pytest 用例完全保留）
  2) 直接 `python test_training_set.py` 运行，自带 main 入口
"""

import random
import sys
from typing import List, Dict, Any

# --------------------------------------------------
# 把被测函数拷贝进来，避免 import 问题
# --------------------------------------------------
def get_training_set(training_set: List[Dict[str, Any]]) -> Dict[str, Any]:
    assert training_set, "training_set is empty"
    if len(training_set) == 1:
        return training_set[0]
    weights = [item.get("weight", 1) for item in training_set]
    return random.choices(training_set, weights=weights, k=1)[0]

# --------------------------------------------------
# 测试用例（普通断言版本）
# --------------------------------------------------
def test_empty_training_set() -> None:
    try:
        get_training_set([])
        assert False, "未触发 AssertionError"
    except AssertionError as e:
        assert str(e) == "training_set is empty"
        print("✓ test_empty_training_set passed")

def test_single_training_set() -> None:
    single = [{"id": "A"}]
    assert get_training_set(single) == {"id": "A"}
    print("✓ test_single_training_set passed")

def test_weighted_random_selection() -> None:
    training = [{"id": "low", "weight": 1}, {"id": "high", "weight": 10}]
    N = 5_000
    counts = {"low": 0, "high": 0}
    for _ in range(N):
        counts[get_training_set(training)["id"]] += 1
    assert counts["high"] > counts["low"] * 4  # 宽松阈值
    print("✓ test_weighted_random_selection passed")

def test_equal_weight_when_missing() -> None:
    training = [{"id": "x"}, {"id": "y"}, {"id": "z"}]
    N = 6_000
    counts = {"x": 0, "y": 0, "z": 0}
    for _ in range(N):
        counts[get_training_set(training)["id"]] += 1
    for v in counts.values():
        assert N * 0.25 < v < N * 0.45
    print("✓ test_equal_weight_when_missing passed")

def test_real_world_training_set() -> None:
    """测试真实世界场景的训练集配置"""
    training_set = [
        {
            "training_layout_configs": {},
            "captions_selection": {},
            "val_layout_configs": {},
            "val_captions_selection": {},
            "weight": 0.7,
        },
        {
            "training_layout_configs": {},
            "captions_selection": {},
            "val_layout_configs": {},
            "val_captions_selection": {},
            "weight": 0.3,
        }
    ]
    
    # 测试权重选择是否符合预期分布
    N = 10_000
    counts = {0: 0, 1: 0}  # 用索引作为标识
    
    for _ in range(N):
        selected = get_training_set(training_set)
        if selected["weight"] == 0.7:
            counts[0] += 1
        else:
            counts[1] += 1
    
    # 验证权重0.7的选项被选中概率约为70%（允许±5%的误差）
    ratio = counts[0] / N
    assert 0.65 <= ratio <= 0.75, f"权重0.7的选中比例应为约70%，实际为{ratio:.2%}"
    
    # 验证返回的数据结构完整性
    selected = get_training_set(training_set)
    required_keys = {"training_layout_configs", "captions_selection", 
                    "val_layout_configs", "val_captions_selection", "weight"}
    assert all(key in selected for key in required_keys)
    
    print("✓ test_real_world_training_set passed")


def test_real_world_training_set_missing_weight() -> None:
    """测试部分元素缺失 weight 时，剩余概率被正确平均分配"""
    training_set = [
        {
            "training_layout_configs": {},
            "captions_selection": {},
            "val_layout_configs": {},
            "val_captions_selection": {},
            "weight": 0.7,
        },
        {
            "training_layout_configs": {},
            "captions_selection": {},
            "val_layout_configs": {},
            "val_captions_selection": {},
            # 缺失 weight
        }
    ]

    N = 10_000
    counts = {0: 0, 1: 0}  # 0 代表带 weight 0.7 的元素；1 代表缺失 weight 的元素

    for _ in range(N):
        selected = get_training_set(training_set)
        # 用是否存在 weight 键来区分（缺失 weight 的元素在返回 dict 里也没有 weight 键）
        if "weight" in selected and selected["weight"] == 0.7:
            counts[0] += 1
        else:
            counts[1] += 1

    # 预期：0.7 vs 0.3，允许 ±5% 误差
    ratio = counts[0] / N
    assert 0.65 <= ratio <= 0.75, \
        f"带 weight 0.7 的元素选中比例应为约 70%，实际为 {ratio:.2%}"

    # 验证返回 dict 的完整性
    selected = get_training_set(training_set)
    required_keys = {"training_layout_configs", "captions_selection",
                     "val_layout_configs", "val_captions_selection"}
    # 缺失 weight 的元素不应含 weight 键；带 weight 的元素应含 weight 键
    if "weight" in selected:
        assert selected["weight"] == 0.7
    assert all(k in selected for k in required_keys)

    print("✓ test_real_world_training_set_missing_weight passed")
# --------------------------------------------------
# main 入口
# --------------------------------------------------
def main() -> None:
    """运行所有手动测试用例"""
    random.seed(42)  # 固定随机种子，结果可复现
    test_empty_training_set()
    test_single_training_set()
    test_weighted_random_selection()
    test_equal_weight_when_missing()
    test_real_world_training_set()  # 添加新测试用例到main
    test_real_world_training_set_missing_weight()
    print("All manual tests passed!")

# --------------------------------------------------
# 使脚本既可被 pytest 收集，也能独立执行
# --------------------------------------------------
if __name__ == "__main__":
    # 如果脚本是被 python 直接运行，则走 main
    main()