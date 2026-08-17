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

## 単元の進め方の型(全単元共通)

1. 該当動画を観る(ユーザー)→ セッションで実装(AI が書いてよい)
2. 参照実装や PyTorch と突き合わせて検証(01 の test_vs_torch 方式)
3. `notes/<単元>.md` に説明を整理 + 前単元からの持ち越し疑問に答えが出たら追記
4. 改造実験を1変数ずつ回して表に記録(予想→実測→考察)
5. 口頭試問 → 通ったら commit & push、CLAUDE.md の「現在地」を更新して次へ

## 各単元でやること

### 01_micrograd(Zero to Hero #1)— スカラー autograd 自作【実質完了】
テーマ: 「forward しながら計算グラフを記録し、逆走して連鎖律で勾配を流す」を
スカラーで最小実装し、PyTorch が裏でやっていることをブラックボックスでなくす。

- **value.py**: Value クラス(data / grad / _prev / _backward)。演算(+, *, **, tanh)
  ごとに「forward 計算 + グラフ記録 + 勾配の配り方」をワンセットで実装。
  backward() はトポロジカルソート → grad=1 → 逆順に _backward 実行
- **nn.py**: Value を組んで Neuron(tanh(w・x+b)、唯一の計算)→ Layer(同じ入力を
  n_out 個に配る)→ MLP(隣接サイズペアで直列)。微分コードゼロ行がポイント
- **train_demo.py**: 4サンプルのおもちゃデータで5拍子
  (forward → loss → zero_grad → backward → update)×100 step。loss 5.28 → 0.011
- **test_vs_torch.py**: 同じ式を PyTorch で組み、forward と勾配の一致を検証(一致済み)
- **experiments.py**: 改造実験6パターンの loss カーブ比較
  (lr=1.0 / tanh なし / zero_grad なし / b=uniform / b=5.0)。
  白眉は zero_grad なし: loss 最良に見えて重みが無限成長する silent bug だった
  (詳細は notes の実験表)

完了条件: 学習ループの各行を説明できる / 勾配が消える停滞(飽和)と発散の違いを
実験で語れる / += と zero_grad の関係を説明できる — いずれも notes に記録済み。
**残タスク: 口頭試問のみ**(出題例: なぜ相手の data を掛ける/なぜ +=/
なぜトポソート/zero_grad を消すとどうなる)

### 02_makemore(Zero to Hero #2〜#6)— 文字レベル言語モデル入門
データ: names.txt(英語の名前 32K 件。makemore リポジトリから取得)。
タスクは一貫して「前の文字(列)から次の文字を当てる」。動画1本 = サブ単元1個:

- **02a bigram(動画#2)**: 頻度カウントで作る最小の言語モデル → 同じものを
  1層 NN + 勾配降下で再現し、両者の loss が一致することを確認。
  cross-entropy / negative log likelihood と sampling が初登場
  (01 の持ち越し疑問「なぜ分類は cross-entropy?」の答え合わせ)。
  実験例: スムージング強度、生成サンプルの質と loss の関係
- **02b MLP(動画#3)**: 文脈3文字 → 埋め込み → 隠れ層 → 次文字予測(Bengio 2003 の縮小版)。
  train/dev/test 分割・ミニバッチ・学習率探索・過学習の観察が初登場。
  実験例: 埋め込み次元 / 文脈長 / 隠れ層サイズを1つずつ変えて dev loss 比較
- **02c 活性化と BatchNorm(動画#4)**: tanh 飽和・初期化スケール(Kaiming)・BatchNorm。
  活性化と勾配のヒストグラムで学習を診断する(01 の「loss 以外も観察する」の本格版)
- **02d backprop ninja(動画#5)**: 02b の MLP の backward を autograd なしで手書きし、
  PyTorch の勾配と一致させる。01 でやったことのテンソル版
- **02e WaveNet 風(動画#6)**: 階層的に文脈を広げる。nn.Module 流のコード整理を覚え、
  03 への橋渡しにする

完了条件: bigram → MLP の loss 改善を表で示せる / cross-entropy を説明できる /
生成サンプルの質を loss と結びつけて語れる

### 03_gpt(Zero to Hero #7 "Let's build GPT")— 自作 Transformer
データ: tiny shakespeare(文字レベル)。bigram ベースラインから始めて部品を1つずつ足す:
self-attention 1ヘッド → multi-head → FFN → ブロック積層 + residual + LayerNorm → スケールアップ。

- 各部品を足すたびに loss がどう動くかを記録する(部品の貢献が見える)
- 成果物: 「attention とは何か」を非エンジニアに説明する1ページ(公開する)
- 完了条件: context length・パラメータ数・バッチサイズを変える実験をして語れる /
  「なぜ attention が必要か」を 02 の MLP の限界(固定長文脈)から説明できる

### 04_nanogpt(karpathy/nanoGPT)— 実物のコードリーディングと日本語学習
- clone して全ファイルを読む(model.py / train.py / sample.py / configurator.py)。
  「地図化」: モジュール構造・データフロー図を notes に自作。03 の自作 GPT との差分
  (実務向けの工夫: AMP・勾配蓄積・checkpoint 再開・torch.compile 等)に注目
- まず shakespeare-char 設定でローカル(MPS)学習 → 動作確認
- 次に日本語コーパス(青空文庫等)で: 文字レベル → トークンレベル(BPE)の順に学習。
  同じテキストの日英トークン数を比較し、日本語のトークン効率問題を数値で押さえる
- 成果物: コードリーディングノート(構造マップ)+ 学習ログ・生成サンプル付き README

### 05_nanochat(karpathy/nanochat)— 読解のみ
学習は回さず、パイプライン全体(tokenizer 学習 → 事前学習 → SFT → 評価 → 推論エンジン → UI)
の構造マップを作る。「事前学習の先に何があるか」の全体像を掴んで卒業。

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
