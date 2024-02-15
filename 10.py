import matplotlib.pyplot as plt
from torch import nn
import torch
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256

w1 = nn.Parameter(
    torch.randn(num_inputs, num_hiddens, requires_grad=True)
)
b1 = nn.Parameter(
    torch.zeros(num_hiddens, requires_grad=True)
)
w2 = nn.Parameter(
    torch.randn(num_hiddens, num_outputs, requires_grad=True)
)
b2 = nn.Parameter(
    torch.zeros(num_outputs, requires_grad=True)
)

params = [w1,b1,w2,b2]

def relu(x):
    a = torch.zeros_like(x)
    return torch.max(x,a)

def net(x):
    x = x.reshape((-1, num_inputs))
    h = relu(x @ w1 + b1)
    return (h @ w2 + b2)


loss = nn.CrossEntropyLoss()
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr = lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
d2l.plt.show()