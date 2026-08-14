"""Value を組み合わせてニューラルネットを作る: Neuron → Layer → MLP。

ここには微分のコードが1行もないことに注目。
forward を Value の演算で書けば、backward はエンジン側が勝手にやってくれる。
これが PyTorch で loss.backward() だけ書けば済む理由と同じ構造。
"""

import random

from value import Value


class Neuron:
    """1ニューロン: tanh(w・x + b)"""

    def __init__(self, n_in):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    """同じ入力を受ける Neuron の束"""

    def __init__(self, n_in, n_out):
        self.neurons = [Neuron(n_in) for _ in range(n_out)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """Layer を直列に重ねた多層パーセプトロン"""

    def __init__(self, n_in, n_outs):
        sizes = [n_in] + n_outs
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(n_outs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
