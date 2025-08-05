import random
from typing import List, Dict, Any
def get_training_set(training_set: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    从 training_set 中选出(或随机选出)一个训练配置。
    当 training_set 为空时抛出 AssertionError。
    当 training_set 只有一个元素时直接返回该元素。
    当 training_set 多于一个元素时：
        - 若元素中存在 'weight' 字段，则按给定权重随机抽取；
        - 若无 'weight'，则将其权重设为剩余概率的平均值，保证所有权重之和为 1。
    """
    # 1. 空列表检查
    assert training_set, "training_set is empty"

    # 2. 只有一个元素直接返回
    if len(training_set) == 1:
        return training_set[0]

    # 3. 多于一个元素：处理权重
    known_total = 0.0
    missing_idx = []
    weights = [0.0] * len(training_set)

    for idx, item in enumerate(training_set):
        w = item.get("weight")
        if w is None:
            missing_idx.append(idx)
        else:
            known_total += float(w)
            weights[idx] = float(w)

    # 保证已知权重之和不超过 1
    assert 0.0 <= known_total <= 1.0, f"Sum of explicit weights must be in [0, 1), got {known_total}"

    # 剩余权重平均分配给缺失 weight 的元素
    if missing_idx:
        leftover = 1.0 - known_total
        avg = leftover / len(missing_idx)
        for idx in missing_idx:
            weights[idx] = avg

    chosen = random.choices(training_set, weights=weights, k=1)[0]
    return chosen