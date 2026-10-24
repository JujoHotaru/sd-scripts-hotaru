# Merge Anima LoRA into a checkpoint / Anima LoRA を checkpoint にマージ

`anima_merge_to_model.py` merges one or more sd-scripts-format Anima DiT LoRAs directly into an Anima `.safetensors` checkpoint. It supports per-LoRA ratios and per-block `--lbws` weights for 28-block Anima and 40-block Anima 2.9B checkpoints.

`anima_merge_to_model.py` は、sd-scripts 形式の Anima DiT LoRA を1つ以上、Anima の `.safetensors` checkpoint に直接マージします。LoRA ごとの ratio と、28層 Anima または40層 Anima 2.9B 向けのブロック別 `--lbws` に対応します。

## Usage / 使用方法

Run from the repository root. Supply one ratio for each LoRA. When `--lbws` is used, supply one JSON array or structured JSON object for each LoRA in the same order. Arrays contain 28 or 40 values and apply to all three Anima sublayers. Structured objects can set separate arrays for `self_attn`, `cross_attn`, and `mlp`.

リポジトリ直下から実行してください。LoRA ごとに ratio を1つ指定します。`--lbws` を使う場合は LoRA と同じ順番で、LoRA ごとに JSON 配列または構造化 JSON を1つ指定します。配列は28個または40個の値を持ち、3つのサブレイヤーすべてに適用されます。構造化 JSON では `self_attn`、`cross_attn`、`mlp` ごとに別の配列を指定できます。

```powershell
python anima_merge_to_model.py `
  --base_model "models\anima.safetensors" `
  --models "loras\style.safetensors" `
  --ratios 0.8 `
  --lbws "[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.5]" `
  --save_to "models\anima_style_merged.safetensors"
```

Use `--lbws_anima` to apply the same additional multiplier to each sublayer across all input LoRAs. It can be used by itself, which gives every block a default weight of `1.0`, or together with one 28- or 40-value `--lbws` array per LoRA:

すべての入力 LoRA の各サブレイヤーに共通の追加倍率を適用するには `--lbws_anima` を使います。単独で指定すると各ブロックの基準倍率は `1.0` です。`--lbws` と併用する場合は、LoRA ごとに28個または40個の配列を指定します。

```powershell
python anima_merge_to_model.py `
  --base_model "models\anima.safetensors" `
  --models "loras\style.safetensors" `
  --ratios 0.8 `
  --lbws_anima "self_attn=1.2,cross_attn=0.8,mlp=0.5" `
  --save_to "models\anima_style_merged.safetensors"
```

The effective scale for a block module is `ratio * lbws[block] * lbws_anima[sublayer]`. The shared sublayer multipliers also apply to matching `adaln_modulation_*` modules. Modules outside numbered blocks keep their normal ratio and are not scaled by `--lbws_anima`.

block module の実効倍率は `ratio * lbws[block] * lbws_anima[sublayer]` です。共通サブレイヤー倍率は対応する `adaln_modulation_*` module にも適用されます。番号付き block の外にある module は通常の ratio を使い、`--lbws_anima` では倍率変更しません。

For multiple LoRAs, pass one `--ratios` value and one `--lbws` value per entry in `--models`. Omit `--lbws` to use weight `1.0` for every block. Modules outside numbered DiT blocks also use weight `1.0`.

複数の LoRA を使う場合、`--models` の各項目に対応する `--ratios` と `--lbws` を指定してください。`--lbws` を省略すると、すべてのブロックを `1.0` として扱います。番号付き DiT block の外にある module も `1.0` です。

## Scope and behavior / 対象範囲と動作

- Merges sd-scripts `lora_unet_*` weights into matching Anima DiT checkpoint weights. Text-encoder LoRA weights are not merged by this script.
- Uses `ratio * alpha / rank * lbws[block] * lbws_anima[sublayer]` as the scale for each block LoRA delta. Both block and shared sublayer weights default to `1.0` when omitted.
- Preserves the base checkpoint metadata, recalculates the safetensors model hashes, and writes the merged checkpoint to `--save_to`.
- Rejects an output path that would overwrite the base checkpoint or an input LoRA.
- Supports safetensors inputs and output. It does not convert checkpoint formats.

- sd-scripts 形式の `lora_unet_*` weights を、対応する Anima DiT checkpoint weight にマージします。このスクリプトは text encoder の LoRA weight をマージしません。
- block LoRA 差分の倍率は `ratio * alpha / rank * lbws[block] * lbws_anima[sublayer]` です。各重みを省略した場合は `1.0` として扱います。
- ベース checkpoint の metadata を保持し、safetensors model hash を再計算してから`--save_to`にマージ済み checkpoint を出力します。
- ベース checkpoint または入力 LoRA を上書きする出力先は指定できません。
- 入出力は safetensors に対応します。checkpoint 形式の変換は行いません。

Use [`anima_merger.py`](../anima_merger.py) when the desired output is a merged LoRA rather than a checkpoint. Use [`anima_lora_scale.py`](../anima_lora_scale.py) to scale one LoRA without merging it into a model.

出力を checkpoint ではなく合成 LoRA にする場合は [`anima_merger.py`](../anima_merger.py)、モデルへマージせず LoRA 1つの倍率だけを変更する場合は [`anima_lora_scale.py`](../anima_lora_scale.py) を使ってください。
