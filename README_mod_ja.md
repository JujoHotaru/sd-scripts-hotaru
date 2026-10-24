# 追加機能を含む sd-scripts

このリポジトリは、[kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) をフォークし、Animaに関するいくつかの学習機能と LoRA ユーティリティを追加したものです。  
インストール方法、依存関係、上流プロジェクトの説明は[原本の日本語 README](./README-ja.md)と[原本の英語 README](./README.md)を参照してください。

[English](./README_mod.md)

## tukisuwa/sd-scriptsから引き継いだ追加機能

本フォークは、[月須和氏のフォーク(tukisuwa/sd-scripts)](https://github.com/tukisuwa/sd-scripts) をベースとし、そちらから以下の機能を継承しています。

- **Anima iLECO 学習** — プロンプトペアを使う Anima 向け学習機能です。[`anima_train_leco.py`](./anima_train_leco.py)と[解説](./docs/anima_train_leco.md)を参照してください。
- **Anima ADDifT 学習** — 画像ペアから Anima 向け edit LoRA を学習します。[`anima_train_addift.py`](./anima_train_addift.py)と[解説](./docs/anima_train_addift.md)を参照してください。
- **アーキテクチャ非依存 LoRA マージ** — 互換性のある sd-scripts 形式の LoRA をマージし、SVD で指定 rank に再構成します。[`merger.py`](./merger.py)と[解説](./docs/merger.md)を参照してください。

## 十条蛍(Hotaru Jujo)による追加・修正

- **Anima LoRA 倍率変更** — SVD を使わず rank を変えずに、`self_attn`、`cross_attn`、`mlp` の強度を変更します。従来の Anima は28層（0〜27）、Anima 2.9B は40層（0〜39）のブロック重みに対応します。[`anima_lora_scale.py`](./anima_lora_scale.py)と[解説](./docs/anima_lora_scale.md)を参照してください。
- **Anima 向け LoRA マージ** — SVD を使って Anima LoRA をマージし、ブロック単位とサブレイヤー単位の倍率を指定できます。[`anima_merger.py`](./anima_merger.py)と[解説](./docs/anima_merger.md)を参照してください。
- **Anima LoRA の checkpoint マージ** — sd-scripts 形式の DiT LoRA をベース checkpoint に直接マージし、LoRA ごとの ratio とブロック重みを指定できます。[`anima_merge_to_model.py`](./anima_merge_to_model.py)と[解説](./docs/anima_merge_to_model.md)を参照してください。
- **メタデータ hash 更新の修正** — tensor データを変更した際に model hash を再計算し、出力メタデータを変更後の tensor に合わせます。

## 注意点

学習機能は実験的です。LoRA ツールでは sd-scripts 形式の互換性がある key と tensor shape が必要です。ブロック重みには従来の Anima 用の28要素配列、または Anima 2.9B 用の40要素配列を指定します。共通モデル読み込み処理には[上流 PR #2418](https://github.com/kohya-ss/sd-scripts/pull/2418)を取り込んでいます。対応範囲と制約は[Anima 2.9B 対応](./docs/anima_29b.md)を参照してください。インストールと基本設定には[原本 README](./README-ja.md)を参照してください。
