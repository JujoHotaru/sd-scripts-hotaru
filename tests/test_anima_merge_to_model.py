"""Anima checkpoint への LoRA merge と prefix 対応を検証する。"""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from safetensors import safe_open
from safetensors.torch import load_file, save_file

import anima_merge_to_model as merge_to_model


@pytest.mark.parametrize("prefix", ["", "net.", "model.diffusion_model."])
@pytest.mark.parametrize("count", [28, 40])
def test_cli_merges_first_and_last_supported_blocks(tmp_path, prefix, count):
    first_block = 0
    last_block = count - 1
    base_weights = {
        f"{prefix}blocks.{first_block}.mlp.layer1.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        f"{prefix}blocks.{last_block}.mlp.layer1.weight": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        f"{prefix}x_embedder.proj.weight": torch.tensor([[9.0, 10.0]]),
    }
    lora_weights = {}
    for block in (first_block, last_block):
        module = f"lora_unet_blocks_{block}_mlp_layer1"
        lora_weights[module + ".lora_down.weight"] = torch.tensor([[3.0, 5.0]])
        lora_weights[module + ".lora_up.weight"] = torch.tensor([[1.0], [2.0]])
        lora_weights[module + ".alpha"] = torch.tensor(2.0)

    base_path = tmp_path / "base.safetensors"
    lora_path = tmp_path / "style.safetensors"
    output_path = tmp_path / "merged.safetensors"
    save_file(base_weights, base_path, metadata={"description": "merge fixture"})
    save_file(lora_weights, lora_path)

    block_weights = [1.0] * count
    block_weights[first_block] = 0.0
    block_weights[last_block] = 0.25
    args = merge_to_model.setup_parser().parse_args(
        [
            "--base_model", str(base_path),
            "--save_to", str(output_path),
            "--models", str(lora_path),
            "--ratios", "0.5",
            "--lbws", json.dumps(block_weights),
            "--save_precision", "fp16",
        ]
    )
    merge_to_model.merge(args)

    saved = load_file(output_path)
    first_key = f"{prefix}blocks.{first_block}.mlp.layer1.weight"
    last_key = f"{prefix}blocks.{last_block}.mlp.layer1.weight"
    untouched_key = f"{prefix}x_embedder.proj.weight"
    assert set(saved) == set(base_weights)
    assert torch.equal(saved[first_key], base_weights[first_key].half())
    torch.testing.assert_close(
        saved[last_key],
        torch.tensor([[5.75, 7.25], [8.5, 10.5]], dtype=torch.float16),
    )
    assert torch.equal(saved[untouched_key], base_weights[untouched_key].half())
    assert all(tensor.dtype == torch.float16 for tensor in saved.values())
    with safe_open(output_path, framework="pt") as handle:
        metadata = handle.metadata()
    assert metadata["description"] == "merge fixture"
    assert len(metadata["sshs_model_hash"]) == 64
    assert len(metadata["sshs_legacy_hash"]) == 8


def test_unsupported_lora_block_is_rejected_before_writing_output(tmp_path):
    base_path = tmp_path / "base.safetensors"
    lora_path = tmp_path / "style.safetensors"
    output_path = tmp_path / "merged.safetensors"
    save_file(
        {
            "blocks.0.mlp.layer1.weight": torch.zeros((2, 2)),
            "blocks.39.mlp.layer1.weight": torch.zeros((2, 2)),
        },
        base_path,
    )
    module = "lora_unet_blocks_40_mlp_layer1"
    save_file(
        {
            module + ".lora_down.weight": torch.ones((1, 2)),
            module + ".lora_up.weight": torch.ones((2, 1)),
        },
        lora_path,
    )
    args = merge_to_model.setup_parser().parse_args(
        [
            "--base_model", str(base_path),
            "--save_to", str(output_path),
            "--models", str(lora_path),
            "--ratios", "0.5",
            "--lbws", json.dumps([1.0] * 28),
        ]
    )

    with pytest.raises(ValueError, match="outside the supported range"):
        merge_to_model.merge(args)
    assert not output_path.exists()


def _expanded_source_index(target):
    """展開先の位置から元の層を算出し、実装とは独立した期待値に使う。"""
    if target == 39:
        return 27
    return (2 * target + (1 if target < 20 else 2)) // 3


def _lora_weights(count, expanded=False):
    state = {}
    for block in range(count):
        value = _expanded_source_index(block) if expanded else block
        module = f"lora_unet_blocks_{block}_mlp_layer1"
        state[module + ".lora_down.weight"] = torch.tensor([[value + 1.0, 1.0]])
        state[module + ".lora_up.weight"] = torch.tensor([[2.0], [value + 2.0]])
        state[module + ".alpha"] = torch.tensor(2.0)
    return state


def _delta(block):
    """rank=1、alpha=2 のテスト用 LoRA の差分を成分ごとに求める。"""
    return torch.tensor([
        [4.0 * (block + 1), 4.0],
        [2.0 * (block + 2) * (block + 1), 2.0 * (block + 2)],
    ])


def _run_depth_merge(tmp_path, source_count, target_count, lbw_count=None, prefix="", shared=False, state=None):
    base = {
        f"{prefix}blocks.{block}.mlp.layer1.weight": torch.full((2, 2), -2.0)
        for block in range(target_count)
    }
    base[f"{prefix}x_embedder.proj.weight"] = torch.tensor([[9.0]])
    base_path = tmp_path / "base.safetensors"
    lora_path = tmp_path / "style.safetensors"
    output_path = tmp_path / "merged.safetensors"
    save_file(base, base_path)
    save_file(_lora_weights(source_count) if state is None else state, lora_path)
    argv = [
        "--base_model", str(base_path), "--models", str(lora_path),
        "--save_to", str(output_path), "--ratios", "0.5",
    ]
    weights = None if lbw_count is None else [block / 20.0 for block in range(lbw_count)]
    if weights is not None:
        argv += ["--lbws", json.dumps(weights)]
    if shared:
        argv += ["--lbws_anima", "self_attn=1.2,cross_attn=0.8,mlp=0.3"]
    merge_to_model.merge(merge_to_model.setup_parser().parse_args(argv))
    return base, load_file(output_path), weights


@pytest.mark.parametrize("prefix", ["", "net.", "model.diffusion_model."])
@pytest.mark.parametrize("lbw_count", [None, 28, 40])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("source_count,target_count", [(28, 40), (40, 28)])
def test_cross_depth_merge_uses_expected_deltas(tmp_path, prefix, lbw_count, shared, source_count, target_count):
    base, saved, weights = _run_depth_merge(tmp_path, source_count, target_count, lbw_count, prefix, shared)
    for target in range(target_count):
        sources = (
            [_expanded_source_index(target)] if source_count == 28
            else [block for block in range(40) if _expanded_source_index(block) == target]
        )
        expected = torch.full((2, 2), -2.0)
        for source in sources:
            lbw = 1.0 if weights is None else weights[source if lbw_count == source_count else target]
            expected += _delta(source) * (0.5 * lbw * (0.3 if shared else 1.0) / len(sources))
        key = f"{prefix}blocks.{target}.mlp.layer1.weight"
        torch.testing.assert_close(saved[key], expected)
    assert set(saved) == set(base)
    assert torch.equal(saved[f"{prefix}x_embedder.proj.weight"], base[f"{prefix}x_embedder.proj.weight"])


def test_reverse_mapping_treats_missing_pair_member_as_zero(tmp_path):
    state = _lora_weights(40)
    for suffix in ("lora_down.weight", "lora_up.weight", "alpha"):
        del state[f"lora_unet_blocks_2_mlp_layer1.{suffix}"]
    _, saved, _ = _run_depth_merge(tmp_path, 40, 28, state=state)
    torch.testing.assert_close(saved["blocks.1.mlp.layer1.weight"], torch.full((2, 2), -2.0) + _delta(1) * 0.25)


def test_reverse_mapping_restores_duplicated_lora_deltas(tmp_path):
    _, saved, _ = _run_depth_merge(tmp_path, 40, 28, state=_lora_weights(40, expanded=True))
    for block in range(28):
        torch.testing.assert_close(saved[f"blocks.{block}.mlp.layer1.weight"], torch.full((2, 2), -2.0) + _delta(block) * 0.5)


def test_forward_mapping_does_not_keep_unmapped_sparse_source_blocks(tmp_path):
    state = _lora_weights(28)
    for key in list(state):
        if not key.startswith(("lora_unet_blocks_1_", "lora_unet_blocks_27_")):
            del state[key]
    _, saved, _ = _run_depth_merge(tmp_path, 28, 40, state=state)
    for target in range(40):
        expected = torch.full((2, 2), -2.0)
        source = _expanded_source_index(target)
        if source in (1, 27):
            expected += _delta(source) * 0.5
        torch.testing.assert_close(saved[f"blocks.{target}.mlp.layer1.weight"], expected)


def test_reverse_mapping_uses_each_contributors_rank_and_alpha(tmp_path):
    state = _lora_weights(40)
    module = "lora_unet_blocks_2_mlp_layer1"
    state[module + ".lora_down.weight"] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    state[module + ".lora_up.weight"] = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    state[module + ".alpha"] = torch.tensor(0.5)
    _, saved, _ = _run_depth_merge(tmp_path, 40, 28, state=state)
    rank_two_delta = torch.tensor([[5.75, 8.5], [7.75, 11.5]])
    expected = torch.full((2, 2), -2.0) + (_delta(1) + rank_two_delta) * 0.25
    torch.testing.assert_close(saved["blocks.1.mlp.layer1.weight"], expected)


@pytest.mark.parametrize("target_count", [28, 40])
def test_mixed_lora_depths_keep_individual_weights_and_counts(tmp_path, target_count, caplog):
    base = {f"blocks.{block}.mlp.layer1.weight": torch.full((2, 2), -2.0) for block in range(target_count)}
    cross_key = f"blocks.{target_count - 1}.cross_attn.k_proj.weight"
    base[cross_key] = torch.zeros((2, 2))
    base_path = tmp_path / "base.safetensors"
    save_file(base, base_path)
    paths = [tmp_path / "small.safetensors", tmp_path / "large.safetensors"]
    save_file(_lora_weights(28), paths[0])
    large = _lora_weights(40)
    module = "lora_unet_blocks_39_cross_attn_k_proj"
    large[module + ".lora_down.weight"] = torch.tensor([[1.0, 2.0]])
    large[module + ".lora_up.weight"] = torch.tensor([[3.0], [4.0]])
    large[module + ".alpha"] = torch.tensor(0.75)
    save_file(large, paths[1])
    small_lbw = [block / 20.0 for block in range(28)]
    large_lbw = [1.0 - block / 80.0 for block in range(40)]
    output_path = tmp_path / "merged.safetensors"
    args = merge_to_model.setup_parser().parse_args([
        "--base_model", str(base_path), "--models", *map(str, paths),
        "--ratios", "0.5", "-0.25", "--save_to", str(output_path),
        "--lbws", json.dumps(small_lbw), json.dumps(large_lbw),
    ])
    merge_to_model.merge(args)
    saved = load_file(output_path)
    for target in range(target_count):
        small_source = _expanded_source_index(target) if target_count == 40 else target
        large_sources = (
            [target] if target_count == 40
            else [block for block in range(40) if _expanded_source_index(block) == target]
        )
        expected = torch.full((2, 2), -2.0) + _delta(small_source) * (0.5 * small_lbw[small_source])
        for source in large_sources:
            expected += _delta(source) * (-0.25 * large_lbw[source] / len(large_sources))
        torch.testing.assert_close(saved[f"blocks.{target}.mlp.layer1.weight"], expected)
    torch.testing.assert_close(
        saved[cross_key], torch.tensor([[3.0, 6.0], [4.0, 8.0]]) * (-0.25 * 0.75 * large_lbw[39])
    )
    assert not any("were not merged" in record.getMessage() for record in caplog.records)
