"""Karpathy の動画と同じ4サンプルのおもちゃデータで MLP を学習する。

学習ループの骨格(これが以後ずっと出てくる5拍子):
  1. forward   : 予測を出す
  2. loss      : 予測と正解の距離を1つのスカラーにまとめる
  3. zero_grad : 前回の勾配を消す(消さないと += で溜まり続ける)
  4. backward  : loss から全パラメータへ勾配を流す
  5. update    : 勾配の逆方向にパラメータを少し動かす
"""

import random

from nn import MLP

random.seed(42)

# 入力3次元 → 4サンプル。目標は [1, -1, -1, 1]
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

model = MLP(3, [4, 4, 1])  # 3 → 4 → 4 → 1
print(f"パラメータ数: {len(model.parameters())}")

lr = 0.05
for step in range(100):
    # 1. forward
    preds = [model(x) for x in xs]
    # 2. loss(二乗誤差の合計)
    loss = sum((p - y) ** 2 for p, y in zip(preds, ys))
    # 3. zero_grad
    for p in model.parameters():
        p.grad = 0.0
    # 4. backward
    loss.backward()
    # 5. update(勾配降下)
    for p in model.parameters():
        p.data -= lr * p.grad

    if step % 10 == 0 or step == 99:
        print(f"step {step:3d} | loss {loss.data:.4f}")

print("\n最終予測 vs 目標:")
for x, y in zip(xs, ys):
    print(f"  pred {model(x).data:+.3f}  target {y:+}")
