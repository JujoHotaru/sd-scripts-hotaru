# sd-scripts with additional features

This repository is a fork of [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) with additional Anima training features and LoRA utilities.
For installation instructions, dependencies, and information about the upstream project, see the [original Japanese README](./README-ja.md) and [original English README](./README.md).

[日本語](./README_mod_ja.md)

## Features inherited from tukisuwa/sd-scripts

This fork is based on [tukisuwa's fork (tukisuwa/sd-scripts)](https://github.com/tukisuwa/sd-scripts) and inherits the following features from it.

- **Anima iLECO training** — Training for Anima using prompt pairs. See [`anima_train_leco.py`](./anima_train_leco.py) and the [guide](./docs/anima_train_leco.md).
- **Anima ADDifT training** — Trains edit LoRAs for Anima from image pairs. See [`anima_train_addift.py`](./anima_train_addift.py) and the [guide](./docs/anima_train_addift.md).
- **Architecture-agnostic LoRA merger** — Merges compatible sd-scripts-style LoRAs and recomposes them at a specified rank using SVD. See [`merger.py`](./merger.py) and the [guide](./docs/merger.md).

## Additions and fixes by Hotaru Jujo

- **Anima LoRA scaling** — Changes the strengths of `self_attn`, `cross_attn`, and `mlp` without SVD or changing the rank. Supports 28-block Anima (0–27) and 40-block Anima 2.9B (0–39). See [`anima_lora_scale.py`](./anima_lora_scale.py) and the [guide](./docs/anima_lora_scale.md).
- **Anima LoRA merger** — Merges Anima LoRAs using SVD, with multipliers for individual blocks and sublayers. See [`anima_merger.py`](./anima_merger.py) and the [guide](./docs/anima_merger.md).
- **Anima LoRA checkpoint merge** — Merges sd-scripts-format DiT LoRAs directly into a base checkpoint, with per-LoRA ratios and block weights. See [`anima_merge_to_model.py`](./anima_merge_to_model.py) and the [guide](./docs/anima_merge_to_model.md).
- **Metadata hash update fix** — Recalculates model hashes when tensor data changes, so the output metadata matches the modified tensors.

## Notes

Training features are experimental. The LoRA tools require compatible sd-scripts-style keys and tensor shapes. Block-weight arrays contain 28 values for Anima or 40 for Anima 2.9B. The common model loader includes [upstream PR #2418](https://github.com/kohya-ss/sd-scripts/pull/2418). See [Anima 2.9B compatibility](./docs/anima_29b.md) for scope and limitations. For installation and basic setup, see the [original README](./README.md).
