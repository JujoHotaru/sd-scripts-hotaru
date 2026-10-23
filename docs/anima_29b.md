# Anima 2.9B 対応

`hotaru_modify_feature_anima29b` は `hotaru_modify` から派生したブランチです。[上流 PR #2418](https://github.com/kohya-ss/sd-scripts/pull/2418)（コミット `45dddfccb704b6b0591f65d98f0d695c997b3115`）をマージし、LoRA の周辺ツールも40層へ対応させています。

## 対応範囲

共通の `load_anima_model()` はチェックポイントのヘッダーにある `blocks.39.mlp.layer1.weight` を検出し、従来の Anima は28層、Anima 2.9B は40層として構築します。`net.`、`model.diffusion_model.`、接頭辞なしのキーを扱います。層数の判定で重み本体を読み込む必要はありません。

One-shot が呼び出す `anima_train_network.py` と ADDifT の `anima_train_addift.py` は、この共通読み込み処理を使います。通常推論、フル学習、LECO、LLLite も共通読み込み処理を利用します。LoRA の生成、ComfyUI 形式への変換、ブロックスワップはモデル内のブロックを動的に扱う構造です。

| ツール | ブロック重み |
| --- | --- |
| `anima_merger.py` | 入力 LoRA ごとに28要素または40要素の配列を指定可能。`--lbws_anima` 単独使用時は入力キーから必要な配列長を決める |
| `anima_lora_scale.py` | 28要素／40要素の配列、または同じ長さのサブレイヤー別配列を指定可能。配列を省略してもブロック39まで倍率変更する |

配列の位置はブロック番号に対応します。ブロック28以降を含む LoRA に28要素だけを指定した場合や、構造化 JSON のサブレイヤー間で配列長が異なる場合はエラーになります。ブロック外のモジュールの扱いは各ツールの既存仕様に従います。

40層すべてを `1.0` にする配列は、PowerShell 7 では次のように作成できます。

```powershell
$blockWeights = ConvertTo-Json -InputObject (@(1.0) * 40) -Compress
python anima_lora_scale.py --input input_29b.safetensors --output scaled_29b.safetensors --lbws $blockWeights
```

学習には通常の BF16 チェックポイントを使用してください。この変更は INT8 ConvRot 形式の読み込みや、新しい量子化方式への対応を追加するものではありません。

## 部分 LoRA とモデル間の互換性

ブロック0〜27だけを含む部分 LoRA から、元モデルが28層か40層かを断定することはできません。既定配列の推定では、その LoRA のキーを扱える最小構成を使います。40要素の配列を明示することもできます。

Anima 2.9B は従来モデルの間に層を挿入した構成です。この変更では、従来の28層用 LoRA を40層モデル向けに変換したり、ブロック番号を対応付け直したりしません。同じブロック番号と重み形状であっても、異なる層数のモデル間で LoRA の意味が一致するとは限りません。

One-shot のコピー機 LoRA は、学習に使ったベースモデルごとにファイル名を分けて管理してください。既存の再利用判定と最終出力のスキップ判定は変更していません。モデルパスや学習条件を変更する処理も、このブランチには含めていません。

## 検証の範囲

回帰テストは CPU 上の小さな LoRA を使い、28層と40層の配列検証、ブロック39の倍率変更・SVD マージ、変換処理、28層／40層の読み込み設定を確認します。

2026-10-23 に関連する既存・新規の CPU 回帰テストを実行し、59件すべて成功しました。CLI 経由の倍率変更・マージ・保存も含みます。独立したコードレビューで指摘はありませんでした。

2026-10-23 に `tools/dev/check_anima_model_headers.py` で手元の通常 BF16 チェックポイントを照合し、28層の `anima-base-v1.0.safetensors` は685テンソル、40層の `Anima-2.9B-preview-v1.safetensors` は928テンソルについてキーと形状が一致しました。モデルは meta device 上で構築し、重み本体や GPU は使用していません。

これらの確認だけでは、実画像を使った One-shot／ADDifT の学習完了、画質、必要 VRAM、学習速度は検証できません。40層モデルでは28層モデルよりメモリと計算量が増えるため、必要に応じて既存のブロックスワップなどを調整してください。

## English

This branch merges upstream PR #2418 and adds 28/40-block support to the Anima merger and scaler. The common loader detects a block-39 key in the checkpoint header. Explicit block-weight arrays must cover every input block, and structured arrays must have equal lengths. Sublayer-only merger defaults are inferred from each input LoRA's keys. Sparse LoRAs do not reliably identify their original architecture. No legacy 28-to-40 block remapping or INT8 ConvRot support is added. CPU regression tests and meta-device header checks do not establish end-to-end training success, image quality, VRAM requirements, or speed.
