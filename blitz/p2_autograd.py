import torch


if __name__ == "__main__":
    x = torch.ones(2, 3, requires_grad=True, dtype=torch.float32)
    y = 3 * x*x
    out = y.mean()
    print(x)
    print(y)
    out.backward()
    print(x.grad)