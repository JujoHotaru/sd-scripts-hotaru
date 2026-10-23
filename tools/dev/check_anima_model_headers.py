"""チェックポイントのヘッダーを meta モデルと照合し、重み本体や GPU を使わず構造を確認する。"""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from library import anima_utils


def check_checkpoint(path):
    with open(path, "rb") as handle:
        header_length = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(header_length))

    def load_meta_weights(**kwargs):
        rename = kwargs["weight_transform_hooks"].rename_hook
        state_dict = {}
        for key, info in header.items():
            if key == "__metadata__":
                continue
            if info["dtype"] != "BF16":
                raise ValueError(f"This check expects BF16 weights: {key} is {info['dtype']}.")
            state_dict[rename(key)] = torch.empty(info["shape"], dtype=torch.bfloat16, device="meta")
        return state_dict

    with torch.device("meta"), patch.object(anima_utils, "load_safetensors_with_lora_and_fp8", load_meta_weights):
        model = anima_utils.load_anima_model("cpu", str(path), "torch", False, "cpu", torch.bfloat16)
    assert len(model.blocks) in (28, 40)
    assert all(parameter.device.type == "meta" for parameter in model.parameters())
    print(f"{Path(path).name}: {len(model.blocks)} blocks, {len(header) - int('__metadata__' in header)} tensors; keys/shapes OK (meta only)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    for checkpoint in parser.parse_args().checkpoints:
        check_checkpoint(checkpoint)
