"""micrograd の核: スカラー値の自動微分エンジン。

Value は1つのスカラーを包み、演算のたびに「計算グラフ」を記録する。
backward() を呼ぶと、グラフを出力側から逆順にたどって連鎖律で勾配を流す。
PyTorch の tensor + autograd がやっていることのスカラー版。
"""

import math


class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data          # 実際の数値(forward の結果)
        self.grad = 0.0           # d(最終出力)/d(この値)。backward() が埋める
        self._backward = lambda: None  # この演算の局所勾配を親から子へ流す関数
        self._prev = set(_children)    # この値を作った入力たち(グラフの辺)
        self._op = _op                 # デバッグ用: どの演算で生まれたか

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # 加算は勾配をそのまま両方の入力へ流す (d(a+b)/da = 1)
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # 乗算は「相手の値」を掛けて流す (d(a*b)/da = b)
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "べき指数は定数のみ対応"
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            # d(x^n)/dx = n * x^(n-1)
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            # d(tanh x)/dx = 1 - tanh^2 x
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        # 1) グラフをトポロジカルソート(子→親の順に並べる)
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # 2) 出力自身の勾配は 1 (d(out)/d(out) = 1)
        self.grad = 1.0

        # 3) 出力側から逆順に、各ノードの局所勾配を流す(連鎖律)
        for v in reversed(topo):
            v._backward()

    # 以下は上の演算の組み合わせで作る便利メソッド
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other ** -1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
