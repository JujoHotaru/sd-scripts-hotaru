import json
import pytest
import torch

pytest.importorskip("torch")
pytest.importorskip("safetensors")

from anima_merger import (
    format_anima_lbw_summary,
    format_anima_lbws_anima_summary,
    get_anima_lbw,
    parse_anima_lbws as parse_merger_anima_lbws,
    parse_anima_lbws_anima,
)
from anima_lora_scale import (
    format_anima_lbw_summary,
    get_anima_sublayer,
    parse_anima_lbws as parse_scale_anima_lbws,
    parse_anima_scale,
    save_lora,
    scale_lora_state_dict,
)


def test_parse_anima_scale_accepts_all_sublayers():
    assert parse_anima_scale("self_attn=2.0,cross_attn=0.5,mlp=0.3") == {
        "self_attn": 2.0,
        "cross_attn": 0.5,
        "mlp": 0.3,
    }


def test_parse_anima_scale_defaults_to_all_ones():
    assert parse_anima_scale(None) == {sublayer: 1.0 for sublayer in ("self_attn", "cross_attn", "mlp")}


def test_save_lora_no_metadata_writes_no_safetensors_metadata(tmp_path):
    output_path = tmp_path / "scaled.safetensors"
    save_lora(output_path, {"test": torch.tensor([1.0])}, no_metadata=True)

    from safetensors import safe_open

    with safe_open(output_path, framework="pt") as handle:
        assert handle.metadata() is None


def test_scale_lora_state_dict_scales_up_weights_only():
    state_dict = {
        "lora_unet_blocks_0_self_attn_to_q.lora_up.weight": torch.tensor([[2.0]]),
        "lora_unet_blocks_0_self_attn_to_q.lora_down.weight": torch.tensor([[3.0]]),
        "lora_unet_blocks_0_cross_attn_k_proj.lora_up.weight": torch.tensor([[4.0]]),
        "lora_unet_blocks_0_cross_attn_k_proj.lora_down.weight": torch.tensor([[5.0]]),
        "lora_unet_blocks_0_mlp_fc1.lora_up.weight": torch.tensor([[6.0]]),
        "lora_unet_blocks_0_mlp_fc1.lora_down.weight": torch.tensor([[7.0]]),
        "lora_unet_x_embedder_proj.lora_up.weight": torch.tensor([[8.0]]),
    }

    scaled, counts, skipped = scale_lora_state_dict(
        state_dict,
        {"self_attn": 2.0, "cross_attn": 0.5, "mlp": 0.3},
    )

    assert scaled["lora_unet_blocks_0_self_attn_to_q.lora_up.weight"].item() == 4.0
    assert scaled["lora_unet_blocks_0_self_attn_to_q.lora_down.weight"].item() == 3.0
    assert scaled["lora_unet_blocks_0_cross_attn_k_proj.lora_up.weight"].item() == 2.0
    assert scaled["lora_unet_blocks_0_mlp_fc1.lora_up.weight"].item() == pytest.approx(1.8)
    assert scaled["lora_unet_x_embedder_proj.lora_up.weight"].item() == 8.0
    assert counts == {"self_attn": 1, "cross_attn": 1, "mlp": 1}
    assert skipped == 0


def test_scale_lora_state_dict_skips_missing_down_weight():
    state_dict = {"lora_unet_blocks_0_mlp_fc1.lora_up.weight": torch.tensor([[1.0]])}

    scaled, counts, skipped = scale_lora_state_dict(state_dict, {sublayer: 2.0 for sublayer in ("self_attn", "cross_attn", "mlp")})

    assert scaled == state_dict
    assert counts == {"self_attn": 0, "cross_attn": 0, "mlp": 0}
    assert skipped == 1


def test_get_anima_sublayer_maps_adaln_and_non_block_modules():
    assert get_anima_sublayer("lora_unet_blocks_2_adaln_modulation_self_attn_0") == "self_attn"
    assert get_anima_sublayer("lora_unet_blocks_2_adaln_modulation_cross_attn_0") == "cross_attn"
    assert get_anima_sublayer("lora_unet_blocks_2_adaln_modulation_mlp_0") == "mlp"
    assert get_anima_sublayer("lora_unet_x_embedder_proj") is None


def test_parse_anima_scale_lbws_uses_one_28_value_array_for_all_sublayers():
    weights = [1.0] * 28
    weights[0] = 2.0

    lbw = parse_scale_anima_lbws([json.dumps(weights)])

    assert lbw["self_attn"][0] == 2.0
    assert lbw["cross_attn"] == weights
    assert lbw["mlp"] == weights


def test_parse_anima_scale_lbws_accepts_structured_sublayer_arrays():
    structured = {
        "self_attn": [2.0] * 28,
        "cross_attn": [0.5] * 28,
        "mlp": [0.3] * 28,
    }

    assert parse_scale_anima_lbws([json.dumps(structured)]) == structured


def test_scale_lora_state_dict_multiplies_scale_and_block_weight():
    state_dict = {
        "lora_unet_blocks_0_self_attn_to_q.lora_up.weight": torch.tensor([[2.0]]),
        "lora_unet_blocks_0_self_attn_to_q.lora_down.weight": torch.tensor([[1.0]]),
        "lora_unet_blocks_27_mlp_fc1.lora_up.weight": torch.tensor([[2.0]]),
        "lora_unet_blocks_27_mlp_fc1.lora_down.weight": torch.tensor([[1.0]]),
    }
    weights = [1.0] * 28
    weights[0] = 2.0
    weights[27] = 3.0

    scaled, counts, skipped = scale_lora_state_dict(
        state_dict,
        {"self_attn": 0.5, "cross_attn": 1.0, "mlp": 2.0},
        parse_scale_anima_lbws([json.dumps(weights)]),
    )

    assert scaled["lora_unet_blocks_0_self_attn_to_q.lora_up.weight"].item() == 2.0
    assert scaled["lora_unet_blocks_27_mlp_fc1.lora_up.weight"].item() == 12.0
    assert counts == {"self_attn": 1, "cross_attn": 0, "mlp": 1}
    assert skipped == 0


def test_scale_lora_state_dict_multiplies_ratio_and_block_weight():
    state_dict = {
        "lora_unet_blocks_0_self_attn_to_q.lora_up.weight": torch.tensor([[2.0]]),
        "lora_unet_blocks_0_self_attn_to_q.lora_down.weight": torch.tensor([[1.0]]),
    }
    weights = [1.0] * 28
    weights[0] = 2.0

    scaled, _, _ = scale_lora_state_dict(
        state_dict,
        {"self_attn": 1.0, "cross_attn": 1.0, "mlp": 1.0},
        parse_scale_anima_lbws([json.dumps(weights)]),
        ratio=0.5,
    )

    assert scaled["lora_unet_blocks_0_self_attn_to_q.lora_up.weight"].item() == 2.0


def test_scale_lora_state_dict_applies_ratio_without_block_weights():
    state_dict = {
        "lora_unet_blocks_0_self_attn_to_q.lora_up.weight": torch.tensor([[2.0]]),
        "lora_unet_blocks_0_self_attn_to_q.lora_down.weight": torch.tensor([[1.0]]),
    }

    scaled, _, _ = scale_lora_state_dict(
        state_dict,
        {"self_attn": 1.0, "cross_attn": 1.0, "mlp": 1.0},
        ratio=0.5,
    )

    assert scaled["lora_unet_blocks_0_self_attn_to_q.lora_up.weight"].item() == 1.0


def test_format_anima_lbw_summary_supports_default_block_weights():
    lbw = {sublayer: [1.0] * 28 for sublayer in ("self_attn", "cross_attn", "mlp")}

    summaries = format_anima_lbw_summary("input.safetensors", 0.5, lbw)

    assert summaries[0] == "Anima strengths for input.safetensors (self_attn): [" + ", ".join(["0.5"] * 28) + "]"
    assert len(summaries) == 3


def test_parse_anima_lbws_anima_accepts_shared_sublayer_multipliers():
    assert parse_anima_lbws_anima("self_attn=2.0,cross_attn=0.5,mlp=0.3") == {
        "self_attn": 2.0,
        "cross_attn": 0.5,
        "mlp": 0.3,
    }


@pytest.mark.parametrize(
    "value",
    [
        "self_attn=2.0,cross_attn=0.5",
        "self_attn=2.0,cross_attn=0.5,mlp=0.3,extra=1",
        "self_attn=2.0,self_attn=0.5,cross_attn=0.5,mlp=0.3",
        "self_attn=bad,cross_attn=0.5,mlp=0.3",
        "",
    ],
)
def test_parse_anima_lbws_anima_rejects_invalid_syntax(value):
    with pytest.raises(ValueError):
        parse_anima_lbws_anima(value)


def test_parse_anima_lbws_rejects_structured_entries_when_multiplier_is_used():
    structured = {sublayer: [1] * 28 for sublayer in ("self_attn", "cross_attn", "mlp")}

    with pytest.raises(ValueError, match="only be used with 28-value JSON array"):
        parse_merger_anima_lbws([json.dumps(structured)], 1, require_arrays=True)


def test_parse_anima_lbws_accepts_default_all_ones_baseline():
    default_lbw = json.dumps([1.0] * 28)

    lbws = parse_merger_anima_lbws([default_lbw, default_lbw], 2, require_arrays=True)

    assert lbws[0]["self_attn"] == [1.0] * 28
    assert lbws[1]["mlp"] == [1.0] * 28


def test_parse_anima_lbws_accepts_one_array_per_model():
    lbws = parse_merger_anima_lbws(["[1, 0.5, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]"], 1)

    assert len(lbws) == 1
    assert len(lbws[0]["self_attn"]) == 28
    assert lbws[0]["self_attn"][1] == 0.5


def test_parse_anima_lbws_accepts_per_sublayer_weights():
    weights = {sublayer: [1.0] * 28 for sublayer in ("self_attn", "cross_attn", "mlp")}
    weights["self_attn"][2] = 0.25
    weights["cross_attn"][2] = 0.5
    weights["mlp"][2] = 0.75

    lbws = parse_merger_anima_lbws([json.dumps(weights)], 1)

    assert lbws[0]["self_attn"][2] == 0.25
    assert lbws[0]["cross_attn"][2] == 0.5
    assert lbws[0]["mlp"][2] == 0.75


@pytest.mark.parametrize(
    "lbws, model_count",
    [
        (["[1]"], 1),
        (["[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \"bad\"]"], 1),
        (["[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]"], 2),
    ],
)
def test_parse_anima_lbws_rejects_invalid_values(lbws, model_count):
    with pytest.raises(ValueError):
        parse_merger_anima_lbws(lbws, model_count)


def test_get_anima_lbw_maps_first_and_last_blocks_only():
    lbw = {sublayer: list(range(28)) for sublayer in ("self_attn", "cross_attn", "mlp")}

    assert get_anima_lbw("lora_unet_blocks_0_self_attn_to_q", lbw) == 0
    assert get_anima_lbw("lora_unet_blocks_27_mlp_fc2", lbw) == 27
    assert get_anima_lbw("lora_unet_x_embedder_proj", lbw) == 1.0


def test_get_anima_lbw_maps_attention_and_adaln_sublayers():
    lbw = {
        "self_attn": [0.1] * 28,
        "cross_attn": [0.2] * 28,
        "mlp": [0.3] * 28,
    }

    assert get_anima_lbw("lora_unet_blocks_4_cross_attn_k_proj", lbw) == 0.2
    assert get_anima_lbw("lora_unet_blocks_4_adaln_modulation_mlp_2", lbw) == 0.3


def test_get_anima_lbw_applies_shared_sublayer_multipliers():
    lbw = {sublayer: [0.5] * 28 for sublayer in ("self_attn", "cross_attn", "mlp")}
    lbws_anima = {"self_attn": 2.0, "cross_attn": 0.5, "mlp": 0.3}

    assert get_anima_lbw("lora_unet_blocks_4_self_attn_to_q", lbw, lbws_anima) == 1.0
    assert get_anima_lbw("lora_unet_blocks_4_cross_attn_k_proj", lbw, lbws_anima) == 0.25
    assert get_anima_lbw("lora_unet_blocks_4_adaln_modulation_mlp_2", lbw, lbws_anima) == 0.15


def test_format_anima_lbw_summary_multiplies_ratio_by_each_sublayer_weight():
    lbw = {
        "self_attn": [0.5] * 28,
        "cross_attn": [1.0] * 28,
        "mlp": [-0.25] * 28,
    }

    summaries = format_anima_lbw_summary("style.safetensors", 2.0, lbw)

    assert summaries[0].startswith("Anima strengths for style.safetensors (self_attn): [")
    assert summaries[0].split("[", 1)[1][:-1].split(", ") == ["1.0"] * 28
    assert summaries[1].split("[", 1)[1][:-1].split(", ") == ["2.0"] * 28
    assert summaries[2].split("[", 1)[1][:-1].split(", ") == ["-0.5"] * 28


def test_format_anima_lbws_anima_summary_preserves_sublayer_order():
    assert format_anima_lbws_anima_summary({"self_attn": 1.0, "cross_attn": 0.5, "mlp": 0.3}) == (
        "self_attn=1.0 / cross_attn=0.5 / mlp=0.3"
    )


def test_get_anima_lbw_rejects_unknown_block_index():
    with pytest.raises(ValueError, match="outside the supported range"):
        get_anima_lbw("lora_unet_blocks_28_self_attn_to_q", {sublayer: list(range(28)) for sublayer in ("self_attn", "cross_attn", "mlp")})