import torch
from torch import nn
from torch.nn import functional as f

#net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(),nn.Linear(256, 10))

x = torch.rand(2, 20)
print(x)
y = torch.rand(size=(2,4))
print(y)
class mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self,x):
        return self.out(f.relu(self.hidden(x)))

#net = mlp()
#a = net(x)
#print(a)

class mysequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for block in args:
            self._modules[block] = block

    def forward(self,x):
        for block in self._modules.values():
            x = block(x)
        return x

#net = mysequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))
#print(net(x))

class fixedhiddenmlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20,20),requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, x):
        x = self.linear(x)
        x = f.relu(torch.mm(x, self.rand_weight) + 1)
        x = self.linear(x)
        while x.abs().sum() > 1:
            x /= 2
        return x.sum()
