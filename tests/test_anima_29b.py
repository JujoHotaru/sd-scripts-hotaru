"""Anima の28層・40層対応を小さなCPU重みで確認する回帰テスト。"""

import json
import runpy
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from safetensors import safe_open
from safetensors.torch import load_file, save_file

import anima_lora_scale as scaler
import anima_merger as merger
from library.anima_block_utils import infer_anima_lora_block_count, validate_anima_block_weights
from networks import convert_anima_lora_to_comfy as converter


SUBLAYERS = ("self_attn", "cross_attn", "mlp")


def _weights(count, value=1.0):
    return {name: [value] * count for name in SUBLAYERS}


def _lora(block):
    base = f"lora_unet_blocks_{block}_mlp_layer1"
    return base, {
        base + ".lora_up.weight": torch.tensor([[2.0], [4.0]]),
        base + ".lora_down.weight": torch.tensor([[3.0, 5.0]]),
        base + ".alpha": torch.tensor(1.0),
    }


def _effect(state_dict, base):
    down = state_dict[base + ".lora_down.weight"]
    return state_dict[base + ".lora_up.weight"] @ down * state_dict[base + ".alpha"] / down.shape[0]


def _merge(path, lbws=None, lbws_anima=None, ratio=1.0, show_lbws=False):
    return merger.merge_lora_models(
        model_paths=[str(path)], ratios=[ratio], lbws=lbws, lbws_anima=lbws_anima,
        new_rank=1, new_conv_rank=None, device="cpu", merge_dtype=torch.float32,
        no_clamp=True, quantize=False, quantize_bits=8, quantize_method="uniform",
        quantize_seed=0, filter_by_first_lora=False, module_filter_rules_parsed=[],
        svd_mode="quality", svd_oversample=0, svd_niter=0, show_lbws=show_lbws,
    )


@pytest.mark.parametrize("count", [28, 40])
@pytest.mark.parametrize("structured", [False, True])
def test_block_arrays_preserve_both_supported_sizes(count, structured):
    expected = _weights(count)
    expected["mlp"][-1] = 3.0
    if structured:
        raw = expected
    else:
        raw = expected["mlp"]
        expected = {name: raw for name in SUBLAYERS}
    encoded = json.dumps(raw)

    assert scaler.parse_anima_lbws([encoded]) == expected
    assert merger.parse_anima_lbws([encoded], 1) == [expected]
    assert merger.get_anima_lbw(f"lora_unet_blocks_{count - 1}_mlp_layer1", expected) == 3.0
    assert validate_anima_block_weights(expected) == count


@pytest.mark.parametrize("count", [27, 29, 39, 41])
def test_unsupported_array_sizes_are_rejected(count):
    raw = json.dumps([1.0] * count)
    with pytest.raises(ValueError):
        scaler.parse_anima_lbws([raw])
    with pytest.raises(ValueError):
        merger.parse_anima_lbws([raw], 1)


def test_mixed_sublayer_lengths_are_rejected():
    raw = _weights(40)
    raw["cross_attn"] = [1.0] * 28
    with pytest.raises(ValueError, match="same length"):
        scaler.parse_anima_lbws([json.dumps(raw)])
    with pytest.raises(ValueError, match="same length"):
        merger.parse_anima_lbws([json.dumps(raw)], 1)


def test_short_weights_reject_block_39_before_scale_or_merge(tmp_path):
    _, state_dict = _lora(39)
    source = tmp_path / "input.safetensors"
    save_file(state_dict, source)
    short = _weights(28)
    with pytest.raises(ValueError, match="supplied 28-value"):
        scaler.scale_lora_state_dict(state_dict, {name: 1.0 for name in SUBLAYERS}, short)
    with pytest.raises(ValueError, match="supplied 28-value"):
        _merge(source, lbws=[short])


def test_block_40_is_rejected_as_outside_supported_models():
    _, state_dict = _lora(40)
    with pytest.raises(ValueError, match="0-39"):
        infer_anima_lora_block_count(state_dict)
    with pytest.raises(ValueError, match="0-39"):
        scaler.scale_lora_state_dict(state_dict, {name: 1.0 for name in SUBLAYERS})


@pytest.mark.parametrize("block", [27, 39])
@pytest.mark.parametrize("use_lbw", [False, True])
def test_scaler_saves_expected_effect_and_preserves_other_weights(tmp_path, block, use_lbw):
    base, state_dict = _lora(block)
    state_dict["lora_unet_x_embedder_proj.lora_up.weight"] = torch.tensor([[7.0]])
    source = tmp_path / "source.safetensors"
    output = tmp_path / "scaled.safetensors"
    save_file(state_dict, source, metadata={"description": "fixture"})
    loaded, metadata = scaler.load_lora(str(source))
    lbw = _weights(block + 1) if use_lbw else None
    if lbw is not None:
        lbw["mlp"][block] = 3.0
    scaled, counts, skipped = scaler.scale_lora_state_dict(
        loaded, {"self_attn": 1.0, "cross_attn": 1.0, "mlp": 2.0}, lbw, ratio=0.5,
    )
    scaler.save_lora(str(output), scaled, metadata)
    saved = load_file(output)

    expected = torch.tensor([[18.0, 30.0], [36.0, 60.0]]) if use_lbw else torch.tensor([[6.0, 10.0], [12.0, 20.0]])
    torch.testing.assert_close(_effect(saved, base), expected)
    assert torch.equal(saved[base + ".lora_down.weight"], state_dict[base + ".lora_down.weight"])
    assert saved[base + ".alpha"].item() == 1.0
    assert saved["lora_unet_x_embedder_proj.lora_up.weight"].item() == 7.0
    assert counts == {"self_attn": 0, "cross_attn": 0, "mlp": 1}
    assert skipped == 0
    with safe_open(output, framework="pt") as handle:
        assert handle.metadata()["description"] == "fixture"


@pytest.mark.parametrize("block", [27, 39])
def test_svd_merger_saves_expected_last_block_effect(tmp_path, block):
    base, state_dict = _lora(block)
    source = tmp_path / "source.safetensors"
    output = tmp_path / "merged.safetensors"
    save_file(state_dict, source)
    lbw = _weights(block + 1)
    lbw["mlp"][block] = 3.0
    merged, metadata = _merge(
        source, lbws=[lbw], ratio=0.5,
        lbws_anima={"self_attn": 1.0, "cross_attn": 1.0, "mlp": 0.5},
    )
    merger.save_to_file(str(output), merged, metadata)
    saved = load_file(output)

    torch.testing.assert_close(_effect(saved, base), torch.tensor([[4.5, 7.5], [9.0, 15.0]]))
    assert saved[base + ".lora_down.weight"].shape == (1, 2)
    assert saved[base + ".lora_up.weight"].shape == (2, 1)
    assert saved[base + ".alpha"].item() == 1.0
    with safe_open(output, framework="pt") as handle:
        assert handle.metadata()["ss_merged_from"] == "source.safetensors"


def test_shared_multiplier_alone_uses_40_default_block_weights(tmp_path, monkeypatch):
    base, state_dict = _lora(39)
    source = tmp_path / "source.safetensors"
    save_file(state_dict, source)
    summaries = []
    original_summary = merger.format_anima_lbw_summary

    def record_summary(path, ratio, lbw):
        summaries.append(lbw)
        return original_summary(path, ratio, lbw)

    monkeypatch.setattr(merger, "format_anima_lbw_summary", record_summary)
    merged, _ = _merge(
        source, lbws_anima={"self_attn": 1.0, "cross_attn": 1.0, "mlp": 0.5},
        ratio=2.0, show_lbws=True,
    )

    assert summaries == [_weights(40)]
    torch.testing.assert_close(_effect(merged, base), torch.tensor([[6.0, 10.0], [12.0, 20.0]]))


def test_comfy_conversion_roundtrip_preserves_block_39(tmp_path):
    base, state_dict = _lora(39)
    source = tmp_path / "source.safetensors"
    comfy = tmp_path / "comfy.safetensors"
    restored = tmp_path / "restored.safetensors"
    save_file(state_dict, source, metadata={"description": "fixture"})
    converter.main(SimpleNamespace(src_path=str(source), dst_path=str(comfy), reverse=False))
    converted = load_file(comfy)
    assert set(converted) == {
        "diffusion_model.blocks.39.mlp.layer1.lora_A.weight",
        "diffusion_model.blocks.39.mlp.layer1.lora_B.weight",
        "diffusion_model.blocks.39.mlp.layer1.alpha",
    }
    torch.testing.assert_close(converted["diffusion_model.blocks.39.mlp.layer1.lora_B.weight"], torch.tensor([[2.0], [4.0]]))
    converter.main(SimpleNamespace(src_path=str(comfy), dst_path=str(restored), reverse=True))
    restored_weights = load_file(restored)
    assert set(restored_weights) == set(state_dict)
    for key, tensor in state_dict.items():
        assert torch.equal(restored_weights[key], tensor)
    torch.testing.assert_close(_effect(restored_weights, base), torch.tensor([[6.0, 10.0], [12.0, 20.0]]))
    with safe_open(restored, framework="pt") as handle:
        assert handle.metadata()["description"] == "fixture"


@pytest.mark.parametrize("prefix", ["", "net.", "model.diffusion_model."])
@pytest.mark.parametrize("count", [28, 40])
def test_loader_detects_checkpoint_depth_before_constructing_real_model(tmp_path, monkeypatch, prefix, count):
    from library import anima_utils

    source = tmp_path / "checkpoint.safetensors"
    save_file({f"{prefix}blocks.{count - 1}.mlp.layer1.weight": torch.tensor([[1.0]])}, source)

    class ModelConstructionReached(Exception):
        pass

    def capture_config(**config):
        # 巨大なモデルの生成より前に停止し、ヘッダーから選択された層数だけを確認する
        raise ModelConstructionReached(config["num_blocks"])

    monkeypatch.setattr(anima_utils.anima_models, "Anima", capture_config)
    with pytest.raises(ModelConstructionReached) as captured:
        anima_utils.load_anima_model("cpu", str(source), "torch", False, "cpu", torch.float32)
    assert captured.value.args == (count,)


@pytest.mark.parametrize("use_lbw", [False, True])
def test_scaler_cli_scales_and_hashes_block_39(tmp_path, monkeypatch, use_lbw):
    base, state_dict = _lora(39)
    source = tmp_path / "source.safetensors"
    output = tmp_path / "scaled.safetensors"
    save_file(state_dict, source, metadata={"description": "fixture"})
    argv = ["anima_lora_scale.py", "--input", str(source), "--output", str(output), "--ratio", "0.5"]
    if use_lbw:
        argv += ["--lbws", json.dumps([2.0] * 40)]
    monkeypatch.setattr(sys, "argv", argv)
    scaler.main()
    expected = torch.tensor([[6.0, 10.0], [12.0, 20.0]]) * (1.0 if use_lbw else 0.5)
    torch.testing.assert_close(_effect(load_file(output), base), expected)
    with safe_open(output, framework="pt") as handle:
        assert handle.metadata()["description"] == "fixture"
        assert len(handle.metadata()["sshs_model_hash"]) == 64


def test_merger_cli_shared_multiplier_alone_covers_block_39(tmp_path, monkeypatch):
    base, state_dict = _lora(39)
    source = tmp_path / "source.safetensors"
    output = tmp_path / "merged.safetensors"
    save_file(state_dict, source)
    monkeypatch.setattr(sys, "argv", [
        "anima_merger.py", "--models", str(source), "--ratios", "1.0", "--new_rank", "1",
        "--save_to", str(output), "--no_clamp", "--save_precision", "float",
        "--lbws_anima", "self_attn=1,cross_attn=1,mlp=0.5",
    ])
    runpy.run_module("anima_merger", run_name="__main__")
    torch.testing.assert_close(_effect(load_file(output), base), torch.tensor([[3.0, 5.0], [6.0, 10.0]]))
