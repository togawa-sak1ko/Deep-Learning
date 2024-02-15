import torch
from torch import nn
from d2l import torch as d2l

def corr2d(x, k):
    h, w = k.shape
    y = torch.zeros((x.shape[0] - h + 1, x.shape[1] - w + 1))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i, j] = (x[i:i+h, j:j+w]*k).sum()
    return y

#x = torch.tensor([[0.0,1.0,2.0],[3.0,4.0,5.0],[6.0,7.0,8.0]])
#y = torch.tensor([[0.0,1.0],[2.0,3.0]])

#print(corr2d(x,y))

class conv2d(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return conv2d(x, self.weight) + self.bias

x = torch.ones((6,8))
x[:,2:6] = 0
k = torch.tensor([[1.0,-1.0]])

y = corr2d(x, k)
#print(y)

conv2d1 = nn.Conv2d(1, 1, kernel_size=(1,2), bias=False)
x = x.reshape((1,1,6,8))
print(x)
y = y.reshape((1,1,6,7))
print(y)

for i in range(10):
    y_hat = conv2d1(x)
    print(i)
    print(y_hat)
    l = (y_hat - y) ** 2
    conv2d1.zero_grad()
    l.sum().backward()
    conv2d1.weight.data[:] -= 3e-2 * conv2d1.weight.grad
    if (i + 1) % 2 == 0:
        print(f'batch {i+1}, loss {l.sum():.3f}')

print(conv2d1.weight.data.reshape((1,2)))