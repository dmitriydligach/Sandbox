#!/usr/bin/env python3

import torch

x = torch.tensor(2.0, requires_grad=True)
y = x**3 + x**2 + 5
z = 2 * y

# dz/dx = 2(3x^2 + 2x) = 2(12 + 4) = 32
z.backward()
print(x.grad.item())
