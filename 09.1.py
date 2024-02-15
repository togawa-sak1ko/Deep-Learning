import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()

trans = transforms.ToTensor()
mnist_trans = torchvision.datasets.FashionMNIST(
    root="../data",train=True,transform=trans,download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data",train=False,transform=trans,download=True)

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt','trouser','pullover','dress','coat',
                   'sandal','shirt','sneaker','bag','ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles = None, scale = 1.5):
    figsize = (num_cols * scale, num_rows *scale)
    _, axes = d2l.plt.subplots(num_rows,num_cols,figsize = figsize)
    axes = axes.flatten()
    for i, (ax,img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

x,y = next(iter(data.DataLoader(mnist_trans ,batch_size=18)))
show_images(x.reshape(18,28,28),2,9,titles=get_fashion_mnist_labels(y))
d2l.plt.show()