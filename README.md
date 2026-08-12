# llm-from-scratch

PyTorch と Transformer をゼロから作って学ぶ練習リポジトリ。
Karpathy「Neural Networks: Zero to Hero」の再実装から nanoGPT / nanochat の読解・改造まで。

## 進め方(AI 併用プロトコル)

コードは AI が書いてもよい。ただし **説明できない状態で次の単元に進まない**。
各単元は以下の3点が揃ったら完了とする:

1. **説明**: コードの各行が何をしているか、`notes/` に自分の言葉で書いた(AI に口頭試問させて詰まらなかった)
2. **改造実験**: パラメータや構造を1変数だけ変えて挙動の変化を観察し、結果を記録した
3. **再現**: 何も見ずに骨格(データ→forward→loss→backward→step)を書き出せる

AI の役割: コード生成・レビュー・口頭試問役・環境整備。
自分の役割: 説明・実験の設計と考察・「なぜ」への回答。

## 構成

```
01_micrograd/   autograd をゼロから(Zero to Hero #1)
02_makemore/    言語モデル入門: bigram → MLP → ... (#2〜#6)
03_gpt/         Let's build GPT: 自作 Transformer (#7)
04_nanogpt/     nanoGPT 読解・日本語コーパスで学習
05_nanochat/    nanochat 読解（SFT・Eval・推論までの全体像）
notes/          単元ごとの説明ノート・実験ログ・構造マップ
```

## 完了条件

- [ ] 学習ループの各行が何をしているか説明できる
- [ ] context length・パラメータ数・バッチサイズを変えると何が起きるか、実験して語れる
- [ ] tokenizer の役割と、日本語で起きる問題(トークン効率)を説明できる

成果物:
- [ ] 自作 GPT(README に構造図・学習ログ・生成サンプル)
- [ ] nanoGPT コードリーディングノート(構造マップ)
- [ ] 「attention とは何か」を非エンジニアに説明する1ページ → 公開する

## 環境

```bash
uv sync
uv run python -c "import torch; print(torch.backends.mps.is_available())"  # True なら GPU(MPS) 使用可
```
