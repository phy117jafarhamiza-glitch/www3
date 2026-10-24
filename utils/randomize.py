# -*- coding: utf-8 -*-
"""توليد تسلسل توزيع عشوائي بكتل متبدلة (Permuted blocks) لثلاث مجموعات، لكل طبقة على حدة."""
import numpy as np

from utils.constants import GROUPS


def permuted_block_sequence(n: int, block_size: int = 6, seed: int = 2026, groups=None):
    groups = groups or GROUPS
    k = len(groups)
    if block_size % k != 0:
        raise ValueError(f"حجم الكتلة يجب أن يقبل القسمة على عدد المجموعات ({k})")
    rng = np.random.default_rng(seed)
    seq = []
    while len(seq) < n:
        block = list(groups) * (block_size // k)
        rng.shuffle(block)
        seq.extend(block)
    return seq[:n]
