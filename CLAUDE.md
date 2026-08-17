# llm-from-scratch — セッション用ガイド

PyTorch と Transformer をゼロから作って学ぶ練習リポジトリ。
Claude はコーチ役。このファイルと `notes/` と `git log` を見れば、どのセッションでも続きから再開できる。

## 現在地(セッション終了時にここを更新すること)

- **進行中の単元**: 01_micrograd(実質完了。残タスク: 口頭試問のみ)
- 01 完了済み: 実装・PyTorch との勾配一致検証・全ファイル解説・改造実験6パターン
  (結果と考察は `notes/01_micrograd.md`)・GitHub 公開
- 次: 口頭試問 → 02_makemore へ

## 学習プロトコル(必ず守る)

コードは AI が書いてよい。ただし**以下3点が揃うまで次の単元に進まない**:

1. **説明**: 各行が何をしているか `notes/<単元>.md` に整理し、口頭試問で詰まらない
2. **改造実験**: 1変数ずつ変えて挙動を観察し、実験表に記録(予想→実測→考察)
3. **再現**: 何も見ずに骨格(データ→forward→loss→backward→update)を言える/書ける

役割分担 — AI: コード生成・解説・口頭試問役・環境整備。ユーザー: 説明・実験の考察・「なぜ」への回答。

コーチングの作法(これまでの経験から):
- 具体例(数字を固定した手計算)+ コードで検算、のセットで解説すると伝わる
- ユーザーの発言が少しずれているときは、合っている部分と直す部分を分けて指摘する
- 「良すぎる結果」が出たら中身を覗く(例: zero_grad なし実験は loss 最良に見えて
  実は重みが無限成長する silent bug だった)

## 各単元でやること

### 01_micrograd(Zero to Hero #1)— 実質完了
スカラー autograd(Value)+ MLP + 学習ループ。PyTorch と勾配一致を検証済み。
残: 口頭試問(出題例は notes と過去ログ参照: なぜ相手の data を掛ける/なぜ +=/
なぜトポソート/zero_grad を消すとどうなる)

### 02_makemore(Zero to Hero #2〜#6)
文字レベル言語モデル入門。names データセットで bigram → MLP → (BatchNorm 等) と段階的に。
- ここで cross-entropy・sampling・train/dev/test 分割が初登場
- 01 から持ち越した疑問(notes 末尾)に答えが出るはず: cross-entropy はなぜ/
  momentum・Adam は何を蓄積している
- 完了条件: bigram と MLP の性能差を loss で比較して語れる

### 03_gpt(Zero to Hero #7 "Let's build GPT")
自作 Transformer。self-attention・位置埋め込み・residual・LayerNorm を1つずつ足す。
- 動画を止めて自分たち(ユーザー+AI)で先に書く→答え合わせ、の再実装スタイル
- 成果物: 「attention とは何か」を非エンジニアに説明する1ページ(公開する)
- 完了条件: context length・パラメータ数・バッチサイズを変える実験をして語れる

### 04_nanogpt(karpathy/nanoGPT)
clone してコードを全部読む(「地図化」: モジュール構造・データフロー図を自作)。
- 改造して日本語コーパス(青空文庫等)でキャラクターレベル→トークンレベル学習
- tokenizer の役割と日本語のトークン効率問題をここで押さえる
- 成果物: コードリーディングノート(構造マップ)+ 学習ログ・生成サンプル付き README

### 05_nanochat(karpathy/nanochat)
読解のみ。SFT・評価・推論サービングまでの全体像を掴む。

## 環境・コマンド

```bash
uv sync                                        # 環境構築
uv run python 01_micrograd/train_demo.py       # 学習デモ
uv run python 01_micrograd/test_vs_torch.py    # PyTorch との勾配一致チェック
uv run python 01_micrograd/experiments.py      # 改造実験6パターンの比較表
```

- Python 3.12 / PyTorch(MPS 可)。GPU 実験はまずローカル MPS → 必要時のみ Colab 等
- 作法: 実験は1変数ずつ / seed 固定で比較 / 「動いた」でなく「測った」まで /
  loss 以外(重み・勾配ノルム)も観察 / 単元完了ごとに commit & push
