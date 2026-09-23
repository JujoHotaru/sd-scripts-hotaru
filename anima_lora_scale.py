import argparse
import json
import logging
import math
import os
import re

import torch
from safetensors.torch import load_file, save_file

ANIMA_LBW_SUBLAYERS = ("self_attn", "cross_attn", "mlp")
ANIMA_LBW_BLOCK_COUNT = 28

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def format_anima_strength(value):
    formatted = f"{value:.6g}"
    if "." not in formatted and "e" not in formatted.lower():
        formatted += ".0"
    return formatted


def format_anima_lbw_summary(input_path, ratio, lbw):
    lines = []
    for sublayer in ANIMA_LBW_SUBLAYERS:
        strengths = [ratio * weight for weight in lbw[sublayer]]
        formatted_strengths = ", ".join(format_anima_strength(strength) for strength in strengths)
        lines.append(f"Anima strengths for {input_path} ({sublayer}): [{formatted_strengths}]")
    return lines


def parse_anima_scale(value):
    if value is None:
        return {sublayer: 1.0 for sublayer in ANIMA_LBW_SUBLAYERS}

    parsed = {}
    for item in value.split(","):
        if not item or "=" not in item:
            raise ValueError("Scale must use self_attn=<number>,cross_attn=<number>,mlp=<number> format.")
        key, raw_scale = item.split("=", 1)
        key = key.strip()
        raw_scale = raw_scale.strip()
        if key not in ANIMA_LBW_SUBLAYERS:
            raise ValueError(f"Unknown scale key '{key}'. Expected: self_attn, cross_attn, mlp.")
        if key in parsed:
            raise ValueError(f"Duplicate scale key '{key}'.")
        try:
            scale = float(raw_scale)
        except ValueError as err:
            raise ValueError(f"Scale value for '{key}' must be a number.") from err
        if not math.isfinite(scale):
            raise ValueError(f"Scale value for '{key}' must be finite.")
        parsed[key] = scale

    missing = [sublayer for sublayer in ANIMA_LBW_SUBLAYERS if sublayer not in parsed]
    if missing:
        raise ValueError(f"Scale is missing required keys: {', '.join(missing)}.")
    return parsed


def parse_anima_lbws(lbws):
    if not lbws:
        return None
    if len(lbws) != 1:
        raise ValueError("The scaler accepts exactly one --lbws entry for its single input LoRA.")

    try:
        parsed_lbw = json.loads(lbws[0])
    except json.JSONDecodeError as err:
        raise ValueError("The --lbws entry must be a JSON array or object.") from err

    if isinstance(parsed_lbw, list):
        if len(parsed_lbw) != ANIMA_LBW_BLOCK_COUNT:
            raise ValueError(
                f"The --lbws array must contain exactly {ANIMA_LBW_BLOCK_COUNT} values for Anima blocks 0-27."
            )
        if not all(isinstance(weight, (int, float)) for weight in parsed_lbw):
            raise ValueError("All --lbws values must be numbers.")
        return {sublayer: parsed_lbw for sublayer in ANIMA_LBW_SUBLAYERS}

    if not isinstance(parsed_lbw, dict):
        raise ValueError(
            "The --lbws entry must be a 28-value JSON array or an object with self_attn, cross_attn, and mlp arrays."
        )
    missing = [sublayer for sublayer in ANIMA_LBW_SUBLAYERS if sublayer not in parsed_lbw]
    if missing:
        raise ValueError(f"Structured --lbws is missing required keys: {', '.join(missing)}.")
    unknown = set(parsed_lbw) - set(ANIMA_LBW_SUBLAYERS)
    if unknown:
        raise ValueError(f"Structured --lbws contains unknown keys: {', '.join(sorted(unknown))}.")

    normalized = {}
    for sublayer in ANIMA_LBW_SUBLAYERS:
        weights = parsed_lbw[sublayer]
        if not isinstance(weights, list) or len(weights) != ANIMA_LBW_BLOCK_COUNT:
            raise ValueError(f"--lbws.{sublayer} must contain exactly {ANIMA_LBW_BLOCK_COUNT} values.")
        if not all(isinstance(weight, (int, float)) for weight in weights):
            raise ValueError(f"All --lbws.{sublayer} values must be numbers.")
        normalized[sublayer] = weights
    return normalized


def get_anima_sublayer(base_module_name):
    match = re.match(r"^lora_unet_blocks_(\d+)_(.+)$", base_module_name)
    if match is None:
        return None

    block_index = int(match.group(1))
    if not 0 <= block_index < 28:
        raise ValueError(f"Anima block index {block_index} in '{base_module_name}' is outside the supported range 0-27.")

    suffix = match.group(2)
    if suffix.startswith("self_attn_") or suffix.startswith("adaln_modulation_self_attn_"):
        return "self_attn"
    if suffix.startswith("cross_attn_") or suffix.startswith("adaln_modulation_cross_attn_"):
        return "cross_attn"
    if suffix.startswith("mlp_") or suffix.startswith("adaln_modulation_mlp_"):
        return "mlp"
    return None


def get_anima_block_index(base_module_name):
    match = re.match(r"^lora_unet_blocks_(\d+)_", base_module_name)
    if match is None:
        raise ValueError(f"Could not determine the Anima block index from '{base_module_name}'.")
    return int(match.group(1))


def load_lora(path):
    if os.path.splitext(path)[1].lower() == ".safetensors":
        with open(path, "rb") as file:
            header_length = int.from_bytes(file.read(8), "little")
            header = json.loads(file.read(header_length))
        return load_file(path), header.get("__metadata__", {})

    loaded = torch.load(path, map_location="cpu")
    if not isinstance(loaded, dict):
        raise ValueError("The input checkpoint must contain a state-dict mapping.")
    return loaded, {}


def save_lora(path, state_dict, metadata=None, no_metadata=False):
    if os.path.splitext(path)[1].lower() == ".safetensors":
        if no_metadata:
            save_file(state_dict, path)
        else:
            save_file(state_dict, path, metadata=metadata)
    else:
        torch.save(state_dict, path)


def scale_lora_state_dict(state_dict, scales, lbw=None, ratio=1.0):
    scaled_modules = {"self_attn": 0, "cross_attn": 0, "mlp": 0}
    skipped_modules = 0
    result = dict(state_dict)

    for key in list(state_dict):
        suffix = ".lora_up.weight"
        if not key.endswith(suffix):
            continue
        base_module_name = key[:-len(suffix)]
        sublayer = get_anima_sublayer(base_module_name)
        if sublayer is None:
            continue
        down_key = base_module_name + ".lora_down.weight"
        if down_key not in state_dict:
            skipped_modules += 1
            logger.warning("Missing '%s'; skipping '%s'.", down_key, base_module_name)
            continue

        scale = scales[sublayer] * ratio
        if lbw is not None:
            block_index = get_anima_block_index(base_module_name)
            scale *= lbw[sublayer][block_index]
        if scale != 1.0:
            result[key] = state_dict[key] * scale
        scaled_modules[sublayer] += 1

    return result, scaled_modules, skipped_modules


def build_parser():
    parser = argparse.ArgumentParser(
        description="Scale self-attention, cross-attention, and MLP modules in an existing Anima LoRA without merging or SVD."
    )
    parser.add_argument("--input", required=True, help="Input Anima LoRA (.safetensors or .pt).")
    parser.add_argument("--output", required=True, help="Output LoRA (.safetensors or .pt).")
    parser.add_argument(
        "--scale",
        default=None,
        help="Sublayer multipliers: self_attn=<number>,cross_attn=<number>,mlp=<number>.",
    )
    parser.add_argument(
        "--lbws",
        type=str,
        nargs="*",
        default=[],
        help="One 28-value JSON array or structured JSON object for Anima blocks 0-27.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=1.0,
        help="Multiplier applied to all --lbws values. Default: 1.0.",
    )
    parser.add_argument(
        "--no_metadata",
        action="store_true",
        help="Do not write any safetensors metadata to the output.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if os.path.abspath(args.input) == os.path.abspath(args.output):
        raise ValueError("Input and output paths must be different.")

    scales = parse_anima_scale(args.scale)
    lbw = parse_anima_lbws(args.lbws)
    if not math.isfinite(args.ratio):
        raise ValueError("The --ratio value must be finite.")
    state_dict, metadata = load_lora(args.input)
    scaled_state_dict, scaled_modules, skipped_modules = scale_lora_state_dict(state_dict, scales, lbw, args.ratio)
    if args.no_metadata:
        output_metadata = None
    else:
        output_metadata = dict(metadata)
        output_metadata["ss_anima_lora_scale"] = json.dumps(scales, sort_keys=True)
        output_metadata["ss_anima_lora_ratio"] = str(args.ratio)
        if args.lbws:
            output_metadata["ss_anima_lora_lbws"] = args.lbws[0]
    save_lora(args.output, scaled_state_dict, output_metadata, args.no_metadata)

    summary_lbw = lbw or {sublayer: [1.0] * ANIMA_LBW_BLOCK_COUNT for sublayer in ANIMA_LBW_SUBLAYERS}
    for summary in format_anima_lbw_summary(args.input, args.ratio, summary_lbw):
        logger.info(summary)
    logger.info("Saved scaled LoRA to %s", args.output)
    logger.info("Scaled modules: %s; skipped incomplete modules: %d", scaled_modules, skipped_modules)


if __name__ == "__main__":
    main()
