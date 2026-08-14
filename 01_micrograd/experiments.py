"""改造実験: train_demo の設定を1変数ずつ変えて loss カーブを比較する。

結果の考察は notes/01_micrograd.md の実験表を参照。
"""

import random

import nn as nn_mod
from nn import MLP

xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]


def fresh():
    """毎回同じ初期重みのモデルを作る(seed 固定 = 実験の比較条件を揃える)"""
    random.seed(42)
    return MLP(3, [4, 4, 1])


def train(model, lr=0.05, zero_grad=True):
    curve = []
    try:
        for step in range(100):
            preds = [model(x) for x in xs]
            loss = sum((p - y) ** 2 for p, y in zip(preds, ys))
            if zero_grad:
                for p in model.parameters():
                    p.grad = 0.0
            loss.backward()
            for p in model.parameters():
                p.data -= lr * p.grad
            if step % 10 == 0:
                curve.append(loss.data)
    except OverflowError:
        pass  # 発散してオーバーフローしたら、そこまでのカーブを返す
    return curve


def fmt(v):
    if v != v:  # nan は自分自身と等しくない
        return "   nan"
    if v >= 1000:
        return f"{v:6.0e}"
    return f"{v:6.3f}"


runs = {}

runs["baseline (lr=0.05)"] = train(fresh())

runs["lr=1.0"] = train(fresh(), lr=1.0)

# tanh を外す(恒等関数に差し替え)
orig = nn_mod.Neuron.__call__
nn_mod.Neuron.__call__ = lambda self, x: sum(
    (wi * xi for wi, xi in zip(self.w, x)), self.b
)
runs["tanh なし"] = train(fresh())
nn_mod.Neuron.__call__ = orig

runs["zero_grad なし"] = train(fresh(), zero_grad=False)

m = fresh()
for layer in m.layers:
    for n in layer.neurons:
        n.b.data = random.uniform(-1, 1)
runs["b=uniform(-1,1)"] = train(m)

m = fresh()
for layer in m.layers:
    for n in layer.neurons:
        n.b.data = 5.0
runs["b=5.0"] = train(m)

header = "step:              " + "".join(f"{s:>7d}" for s in range(0, 100, 10))
print(header)
print("-" * len(header))
for name, curve in runs.items():
    print(f"{name:19s}" + "".join(f" {fmt(v)}" for v in curve))
