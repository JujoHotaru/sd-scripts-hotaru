# Fast Anima LoRA Sublayer Scaling / Anima LoRA サブレイヤー倍率変更

`anima_lora_scale.py` changes the strength of all Anima `self_attn`, `cross_attn`, and `mlp` LoRA modules without merging, SVD, or changing the LoRA rank. It scales only `lora_up.weight`, so `lora_down.weight`, `alpha`, and the LoRA structure remain unchanged.

`anima_lora_scale.py` は、マージや SVD を行わずに、Anima LoRA の `self_attn`、`cross_attn`、`mlp` 各層の倍率だけを高速に変更します。`lora_up.weight` のみを変更するため、`lora_down.weight`、`alpha`、rank、LoRA の構造は維持されます。

## Requirements / 前提

- Run the command from the repository root.
- `safetensors` and `torch` must be available.
- The input should be an Anima LoRA using sd-scripts style `lora_down` / `lora_up` keys.
- Input and output paths must be different.

- リポジトリ直下から実行してください。
- `safetensors` と `torch` が必要です。
- 入力は sd-scripts 形式の `lora_down` / `lora_up` キーを持つ Anima LoRA を指定してください。
- 入力と出力には異なるパスを指定してください。

## Usage / 使用方法

```powershell
python anima_lora_scale.py `
  --input input_anima.safetensors `
  --output output_anima.safetensors `
  --scale "self_attn=2.0,cross_attn=0.5,mlp=0.3"
```

The `--scale` switch is optional. When omitted, `self_attn`, `cross_attn`, and `mlp` all use `1.0`.

`--scale` は省略できます。省略した場合、`self_attn`、`cross_attn`、`mlp` の倍率はすべて `1.0` になります。

The `--ratio` switch is optional and defaults to `1.0`. It is applied as a multiplier to every `--lbws` value:

`--ratio` は省略可能で、既定値は `1.0` です。`--lbws` の各値に対する共通倍率として適用されます。

```powershell
python anima_lora_scale.py `
  --input input_anima.safetensors `
  --output output_anima.safetensors `
  --ratio 0.5 `
  --lbws "[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.8]"
```

To change individual DiT block strengths, use one `--lbws` entry with 28 values for Anima (blocks 0–27), or 40 values for Anima 2.9B (blocks 0–39). The positions map directly to block indices, matching `anima_merger.py`. The following example uses a 28-block LoRA:

DiT ブロックごとの強度を変更する場合は、`--lbws` に従来の Anima 用の28要素（0〜27）、または Anima 2.9B 用の40要素（0〜39）の JSON 配列を1つ指定します。配列の位置は `anima_merger.py` と同じくブロック番号に対応します。次の例は28層の LoRA 用です。

```powershell
python anima_lora_scale.py `
  --input input_anima.safetensors `
  --output output_anima.safetensors `
  --scale "self_attn=2.0,cross_attn=0.5,mlp=0.3" `
  --lbws "[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.5]"
```

The structured JSON form is also supported when different block weights are needed for each sublayer:

サブレイヤーごとに異なるブロック倍率が必要な場合は、構造化 JSON 形式も使用できます。

```powershell
python anima_lora_scale.py `
  --input input_anima.safetensors `
  --output output_anima.safetensors `
  --scale "self_attn=1.0,cross_attn=1.0,mlp=1.0" `
  --lbws '{"self_attn":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.8],"cross_attn":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.6],"mlp":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.4]}'
```

When both switches are specified, the effective multiplier is multiplied per module:

両方を指定した場合、各 module への実効倍率は次のように乗算されます。

```text
effective multiplier = --scale[sublayer] * --ratio * --lbws[sublayer][block_index]
```

The scaler prints 28 or 40 combined `--ratio * --lbws` values for each sublayer using the same `Anima strengths for ...` format as `anima_merger.py`. Without `--lbws`, the summary length is inferred from the input block indices.

`anima_merger.py` と同じ `Anima strengths for ...` 形式で、サブレイヤーごとの `--ratio * --lbws` の合成値28個または40個をコンソールに出力します。`--lbws` を省略した場合は入力のブロック番号から表示する層数を判断します。

An array applies the same 28 or 40 weights to all three sublayers. Structured arrays must have matching lengths. Omitting `--lbws` applies `1.0` to every block. A 28-value array cannot cover an input containing block 28 or later; this fails before saving.

配列形式では、同じ28要素または40要素のブロック倍率を3つのサブレイヤーへ適用します。構造化 JSON の3つの配列も同じ長さにそろえます。`--lbws` を省略した場合はすべて `1.0` として扱います。ブロック28以降を含む入力に28要素だけを指定すると、保存前にエラーになります。

The `--scale` value must contain all three keys exactly once:

`--scale` には、次の 3 キーをそれぞれ 1 回ずつ指定する必要があります。

```text
self_attn=<number>,cross_attn=<number>,mlp=<number>
```

The values may be positive, zero, or negative finite numbers. For example:

値には、正数、`0`、負数の有限な数値を指定できます。例:

```powershell
python anima_lora_scale.py `
  --input input_anima.safetensors `
  --output output_anima.safetensors `
  --scale "self_attn=1.2,cross_attn=0.8,mlp=0.5"
```

## Target Modules / 対象モジュール

The multipliers are applied to `lora_up.weight` for matching modules:

倍率は、対応する module の `lora_up.weight` に適用されます。

- `self_attn`: `self_attn_*` and `adaln_modulation_self_attn_*`
- `cross_attn`: `cross_attn_*` and `adaln_modulation_cross_attn_*`
- `mlp`: `mlp_*` and `adaln_modulation_mlp_*`

The tool supports Anima blocks 0–27 and Anima 2.9B blocks 0–39. Non-block modules such as `x_embedder`, `t_embedder`, `final_layer`, and `llm_adapter` are left unchanged. It does not remap block indices between architectures.

従来の Anima のブロック0〜27と、Anima 2.9B のブロック0〜39に対応しています。`x_embedder`、`t_embedder`、`final_layer`、`llm_adapter` など、ブロック外のモジュールは変更されません。28層と40層の間でブロック番号を変換する処理はありません。[対応範囲と制約](./anima_29b.md)も確認してください。

## Preservation / 保持されるもの

The tool does not merge LoRAs or perform SVD. It preserves:

このツールは LoRA のマージや SVD を行いません。次の要素を保持します。

- `lora_down.weight`
- `.alpha`
- rank and tensor shapes
- non-target tensors
- existing safetensors metadata, unless `--no_metadata` is used

Only the matching `lora_up.weight` tensors are multiplied. The output metadata additionally records the requested values in `ss_anima_lora_scale` and `ss_anima_lora_ratio`. When `--lbws` is specified, its JSON value is also recorded in `ss_anima_lora_lbws`.

対応する `lora_up.weight` tensor だけを倍率変更します。通常の出力 metadata には、指定した値が `ss_anima_lora_scale` と `ss_anima_lora_ratio` として追加記録されます。`--lbws` を指定した場合は、JSON の値も `ss_anima_lora_lbws` に記録されます。

Use `--no_metadata` to write a safetensors file without metadata at all. Existing input metadata and the scaler's own metadata are both omitted.

`--no_metadata` を指定すると、メタデータを一切持たない safetensors を出力します。入力 LoRA の既存 metadata と、スケーラーが追加する metadata の両方が出力されません。

```powershell
python anima_lora_scale.py `
  --input input_anima.safetensors `
  --output output_anima.safetensors `
  --no_metadata
```

## When to Use / 使い分け

Use this tool when you want to adjust the relative strength of Anima sublayers in an existing LoRA without changing its rank or combining it with another LoRA.

既存 LoRA の rank を変更せず、別の LoRA と合成することもなく、Anima のサブレイヤーごとの強度だけを調整したい場合に使用します。

Use `anima_merger.py` when you need to combine multiple LoRAs or reconstruct the result at a new rank with SVD.

複数 LoRA の合成や、SVD による新しい rank への再構成が必要な場合は `anima_merger.py` を使用します。
