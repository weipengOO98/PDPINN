import torch
import abc
from collections import OrderedDict
from torch import cos, sin


class MLP(torch.nn.Module):
    def __init__(self, seq, name='mlp'):
        super().__init__()
        self.layers = OrderedDict()
        for i in range(len(seq) - 1):
            self.layers['{}_{}'.format(name, i)] = torch.nn.Linear(seq[i], seq[i + 1])
        self.layers = torch.nn.ModuleDict(self.layers)

    def forward(self, x):
        l = len(self.layers)
        i = 0
        for name, layer in self.layers.items():
            x = layer(x)
            if i == l - 1: break
            i += 1
            x = torch.tanh(x)
        return x


class MLP_sin(MLP):
    def forward(self, x):
        l = len(self.layers)
        i = 0
        for name, layer in self.layers.items():
            x = layer(x)
            if i == l - 2: break
            x = torch.sin(x)
        return x


class Net(torch.nn.Module):
    def __init__(self, seq):
        super().__init__()
        self.mlp = MLP(seq)

    def forward(self, x):
        return self.mlp(x)


class Basis_Net(torch.nn.Module):
    def __init__(self, seq, basis):
        super().__init__()
        self.basis_num = seq[-1]
        self.mlp = MLP(seq)
        self.basis = basis

    def forward(self, x):
        s = self.basis(x)
        x = self.mlp(x)
        return (x * s).sum(1).reshape((-1, 1))


class Basis_Net_Time(Basis_Net):
    def forward(self, x):
        s = self.basis(x[:, :1])
        x = self.mlp(x)
        return (x * s).sum(1).reshape((-1, 1))


class Sphere_Net(Net):
    def forward(self, x):
        x = coordinates_get_3d(x)
        x = self.mlp(x)
        return x


class SPH_Sphere_Net(Basis_Net):
    def forward(self, x):
        s = self.basis(x)
        x = coordinates_get_3d(x)
        x = self.mlp(x)
        x = (x * s).sum(dim=1).reshape((-1, 1))
        return x


def coordinates_get_3d(x):
    u = torch.sin(x[:, :1]) * torch.sin(x[:, 1:])
    v = torch.sin(x[:, :1]) * torch.cos(x[:, 1:])
    w = torch.cos(x[:, :1])
    x = torch.cat([u, v, w], dim=1)
    return x
