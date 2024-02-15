import torch
from torch import nn
from d2l import torch as d2l

n_train, n_test, num_inputs, batch_size = 20,100,200,5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
#print(true_w, true_b)
train_data = d2l.synthetic_data(true_w, true_b, n_train)
#print(train_data)
train_iter = d2l.load_array(train_data, batch_size)
#print(train_iter)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data,batch_size,is_train=False)

def init_params():
    w = torch.normal(0, 1, size=(num_inputs,1),requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

def l2_penalty(w):
    return torch.sum(w.pow(2))/2

