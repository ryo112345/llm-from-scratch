"""自作 autograd の勾配が PyTorch と一致するか検証する。

「動いた」でなく「測った」: 同じ計算式を両方で組み、勾配を突き合わせる。
"""

import torch

from value import Value


def micrograd_grads():
    a = Value(-2.0)
    b = Value(3.0)
    c = a * b + b ** 2          # -6 + 9 = 3
    d = c.tanh() + a / b        # 複数演算を混ぜる
    e = (d + 1.0) * (d - 0.5)
    e.backward()
    return e.data, a.grad, b.grad


def torch_grads():
    a = torch.tensor(-2.0, requires_grad=True)
    b = torch.tensor(3.0, requires_grad=True)
    c = a * b + b ** 2
    d = torch.tanh(c) + a / b
    e = (d + 1.0) * (d - 0.5)
    e.backward()
    return e.item(), a.grad.item(), b.grad.item()


if __name__ == "__main__":
    (e1, ga1, gb1) = micrograd_grads()
    (e2, ga2, gb2) = torch_grads()
    print(f"forward : micrograd {e1:.6f} | torch {e2:.6f}")
    print(f"da      : micrograd {ga1:.6f} | torch {ga2:.6f}")
    print(f"db      : micrograd {gb1:.6f} | torch {gb2:.6f}")
    assert abs(e1 - e2) < 1e-6 and abs(ga1 - ga2) < 1e-6 and abs(gb1 - gb2) < 1e-6
    print("OK: 勾配が PyTorch と一致")
