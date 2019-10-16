#!/usr/bin/env python3

import torch

x = torch.tensor(2.0, requires_grad=True)
y = x**3 + x**2 + 5
z = 2 * y

# dz/dx
z.backward()
print(x.grad)

