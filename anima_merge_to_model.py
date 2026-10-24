import argparse
import logging
import os

import torch
from tqdm import tqdm

from anima_merger import (
    format_anima_lbw_summary,
    format_anima_lbws_anima_summary,
    get_anima_lbw,
    parse_anima_lbws,
    parse_anima_lbws_anima,
)
from library.anima_block_utils import infer_anima_lora_block_count, validate_anima_block_weights
from library import model_io
from library.safetensors_utils import MemoryEfficientSafeOpen, mem_eff_save_file
from library.utils import setup_logging, str_to_dtype

setup_logging()
logger = logging.getLogger(__name__)


def merge_single_lora_weight(weight, lora_down, lora_up, alpha, ratio):
    rank = lora_down.size(0)
    scale = ratio * (alpha / rank)

    if weight.ndim == 2:
        if lora_up.ndim == 4:
            lora_up = lora_up.squeeze(3).squeeze(2)
            lora_down = lora_down.squeeze(3).squeeze(2)
        delta = lora_up @ lora_down
    elif lora_down.shape[2:4] == (1, 1):
        delta = (lora_up.squeeze(3).squeeze(2) @ lora_down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
    else:
        delta = torch.nn.functional.conv2d(lora_down.permute(1, 0, 2, 3), lora_up).permute(1, 0, 2, 3)

    if delta.shape != weight.shape:
        raise ValueError(f"LoRA delta shape {tuple(delta.shape)} does not match model weight shape {tuple(weight.shape)}.")
    return weight + scale * delta


def build_anima_lora_mapping(lora_keys, model_keys):
    """Anima の checkpoint key と sd-scripts LoRA module 名を対応付ける。"""
    model_modules = {}
    for key in model_keys:
        if not key.endswith(".weight"):
            continue
        base = key[:-len(".weight")]
        if base.startswith("net."):
            base = base[len("net.") :]
        model_modules.setdefault(base.replace(".", "_"), key)

    module_names = {
        key.split(".lora_down.", 1)[0]
        for key in lora_keys
        if ".lora_down." in key and key.startswith("lora_unet_")
    }
    mapping = {}
    unmatched = []
    for module_name in sorted(module_names):
        flattened_name = module_name[len("lora_unet_") :]
        model_key = model_modules.get(flattened_name)
        if model_key is None:
            unmatched.append(module_name)
        else:
            mapping[model_key] = module_name

    if unmatched:
        logger.warning("%d Anima LoRA modules did not match the base model; examples: %s", len(unmatched), unmatched[:5])
    return mapping, len(module_names), len(unmatched)


def merge_lora_weights(key, base_weight, lora_files, lora_keys, mappings, ratios, lbws, lbws_anima, merge_dtype, device):
    if not key.endswith(".weight"):
        return [False] * len(lora_files), base_weight

    successes = []
    for lora_file, keys, mapping, ratio, lbw in zip(lora_files, lora_keys, mappings, ratios, lbws):
        module_name = mapping.get(key)
        if module_name is None:
            successes.append(False)
            continue

        down_key = module_name + ".lora_down.weight"
        up_key = module_name + ".lora_up.weight"
        if down_key not in keys or up_key not in keys:
            successes.append(False)
            continue

        down = lora_file.get_tensor(down_key, device=device, dtype=merge_dtype)
        up = lora_file.get_tensor(up_key, device=device, dtype=merge_dtype)
        alpha_key = module_name + ".alpha"
        alpha = float(lora_file.get_tensor(alpha_key, device=device, dtype=merge_dtype)) if alpha_key in keys else down.size(0)
        module_ratio = ratio * get_anima_lbw(module_name, lbw, lbws_anima)
        base_weight = merge_single_lora_weight(base_weight, down, up, alpha, module_ratio)
        successes.append(True)

    return successes, base_weight


def merge(args):
    if len(args.models) != len(args.ratios):
        raise ValueError("The number of --models and --ratios entries must match.")
    if args.lbws and len(args.lbws) != len(args.models):
        raise ValueError("Specify one --lbws entry for each --models entry.")
    output_path = os.path.normcase(os.path.abspath(args.save_to))
    input_paths = [args.base_model, *args.models]
    if any(output_path == os.path.normcase(os.path.abspath(path)) for path in input_paths):
        raise ValueError("--save_to must be different from the base model and every input LoRA.")

    lbws_anima = parse_anima_lbws_anima(args.lbws_anima)
    lbws = parse_anima_lbws(args.lbws, len(args.models), require_arrays=lbws_anima is not None)
    if lbws_anima is not None:
        logger.info("Anima shared sublayer strengths: %s", format_anima_lbws_anima_summary(lbws_anima))

    merge_dtype = str_to_dtype(args.precision)
    save_dtype = str_to_dtype(args.save_precision) if args.save_precision else merge_dtype
    device = torch.device(args.device) if args.device else torch.device("cpu")

    logger.info("Checking checkpoint and LoRA keys...")
    with MemoryEfficientSafeOpen(args.base_model) as base_file:
        model_keys = base_file.keys()
        base_metadata = base_file.metadata()

    lora_keys = []
    mappings = []
    module_counts = []
    effective_lbws = []
    for i, model_path in enumerate(args.models):
        with MemoryEfficientSafeOpen(model_path) as lora_file:
            keys = lora_file.keys()
        if not any(".lora_down." in key for key in keys):
            raise ValueError(f"No sd-scripts lora_down weights found in '{model_path}'.")
        if lbws is None and lbws_anima is not None:
            lbw = {sublayer: [1.0] * infer_anima_lora_block_count(keys) for sublayer in ("self_attn", "cross_attn", "mlp")}
        elif lbws is None:
            lbw = None
        else:
            lbw = lbws[i]
        if lbw is not None:
            validate_anima_block_weights(lbw, keys)
        mapping, matched_module_count, unmatched_module_count = build_anima_lora_mapping(keys, model_keys)
        if matched_module_count == 0 or matched_module_count == unmatched_module_count:
            raise ValueError(f"No Anima DiT LoRA modules matched the base model in '{model_path}'.")
        lora_keys.append(set(keys))
        mappings.append(mapping)
        module_counts.append(matched_module_count - unmatched_module_count)
        effective_lbws.append(lbw)
        logger.info(
            "LoRA %d: matched %d of %d modules (%d unmatched)",
            i,
            matched_module_count - unmatched_module_count,
            matched_module_count,
            unmatched_module_count,
        )
        if lbw is not None:
            for line in format_anima_lbw_summary(model_path, args.ratios[i], lbw):
                logger.info(line)

    lbws = effective_lbws

    merged_sd = {}
    with MemoryEfficientSafeOpen(args.base_model) as base_file:
        lora_files = [MemoryEfficientSafeOpen(path) for path in args.models]
        try:
            for key in tqdm(base_file.keys()):
                value = base_file.get_tensor(key, device=device, dtype=merge_dtype)
                success_list, value = merge_lora_weights(
                    key, value, lora_files, lora_keys, mappings, args.ratios, lbws, lbws_anima, merge_dtype, device
                )
                merged_sd[key] = value.to(device="cpu", dtype=save_dtype)
                for i, success in enumerate(success_list):
                    if success:
                        module_counts[i] -= 1
        finally:
            for lora_file in lora_files:
                lora_file.__exit__(None, None, None)

    if any(count > 0 for count in module_counts):
        logger.warning("Some mapped LoRA modules were not merged: %s", module_counts)
    else:
        logger.info("All matched Anima DiT LoRA modules were merged.")

    save_dir = os.path.dirname(os.path.abspath(args.save_to))
    os.makedirs(save_dir, exist_ok=True)
    logger.info("Saving merged model to: %s", args.save_to)
    if base_metadata is None:
        base_metadata = {}
    else:
        base_metadata = dict(base_metadata)
    base_metadata.pop("sshs_model_hash", None)
    base_metadata.pop("sshs_legacy_hash", None)
    mem_eff_save_file(merged_sd, args.save_to, metadata=base_metadata)
    with open(args.save_to, "rb") as output_file:
        model_hash = model_io.addnet_hash_safetensors(output_file)
        legacy_hash = model_io.addnet_hash_legacy(output_file)
    base_metadata["sshs_model_hash"] = model_hash
    base_metadata["sshs_legacy_hash"] = legacy_hash
    mem_eff_save_file(merged_sd, args.save_to, metadata=base_metadata)


def setup_parser():
    parser = argparse.ArgumentParser(description="Merge sd-scripts Anima LoRA weights into an Anima checkpoint.")
    parser.add_argument("--base_model", required=True, help="Anima base checkpoint in safetensors format.")
    parser.add_argument("--save_to", required=True, help="Output merged checkpoint in safetensors format.")
    parser.add_argument("--models", nargs="+", required=True, help="Anima sd-scripts LoRA safetensors files.")
    parser.add_argument("--ratios", nargs="+", type=float, required=True, help="Multiplier for each LoRA.")
    parser.add_argument(
        "--lbws",
        nargs="*",
        default=[],
        help="Optional per-LoRA JSON block weights: 28/40 values, or per-sublayer arrays.",
    )
    parser.add_argument(
        "--lbws_anima",
        default=None,
        help="Shared Anima sublayer multipliers: self_attn=<number>,cross_attn=<number>,mlp=<number>.",
    )
    parser.add_argument("--precision", choices=["float", "fp16", "bf16"], default="float")
    parser.add_argument("--save_precision", choices=["float", "fp16", "bf16"], default=None)
    parser.add_argument("--device", default="cpu", help="Merge device, such as cpu or cuda.")
    return parser


if __name__ == "__main__":
    merge(setup_parser().parse_args())
