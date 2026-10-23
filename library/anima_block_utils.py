"""Anima 28層・40層 LoRA のブロック重みを検証する共通処理。"""

import re


ANIMA_BLOCK_COUNTS = (28, 40)
ANIMA_SUBLAYERS = ("self_attn", "cross_attn", "mlp")


def infer_anima_lora_block_count(keys):
    """存在するブロック番号から必要な層数を求め、部分 LoRA は最小構成として扱う。"""
    last_block = -1
    for key in keys:
        match = re.match(r"^lora_unet_blocks_(\d+)_", key)
        if match is not None:
            last_block = max(last_block, int(match.group(1)))
    for count in ANIMA_BLOCK_COUNTS:
        if last_block < count:
            return count
    raise ValueError(f"Anima block index {last_block} is outside the supported range 0-39.")


def validate_anima_block_weights(lbw, keys=()):
    """全サブレイヤーの配列長と、入力に含まれるブロックの範囲を確認する。"""
    counts = set()
    for sublayer in ANIMA_SUBLAYERS:
        weights = lbw[sublayer]
        if not isinstance(weights, list) or len(weights) not in ANIMA_BLOCK_COUNTS:
            raise ValueError(f"--lbws.{sublayer} must contain exactly 28 or 40 values.")
        if not all(isinstance(weight, (int, float)) for weight in weights):
            raise ValueError(f"All --lbws.{sublayer} values must be numbers.")
        counts.add(len(weights))
    if len(counts) != 1:
        raise ValueError("All --lbws sublayer arrays must have the same length (28 or 40).")
    count = counts.pop()
    for key in keys:
        match = re.match(r"^lora_unet_blocks_(\d+)_", key)
        if match is not None and int(match.group(1)) >= count:
            raise ValueError(
                f"Anima block index {match.group(1)} in '{key}' is outside the supported range 0-{count - 1} "
                f"for the supplied {count}-value --lbws. Use 40 values for a 40-block LoRA."
            )
    return count
