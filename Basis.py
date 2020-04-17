import torch
import abc
from math import pi
from torch import cos, sin
import matplotlib.pyplot as plt
import seaborn as sns
import math


class Basis(torch.nn.Module, metaclass=abc.ABCMeta):
    def __int__(self):
        super(Basis, self).__init__()
        pass

    @abc.abstractmethod
    def forward(self, x):
        pass

    def get_plot(self):
        pass

    def __str__(self):
        pass


class FourierBasis1D(Basis):
    def __init__(self, basis_num: int, l: float, u: float):
        assert basis_num % 2 == 1
        super().__init__()
        self.basis_num = basis_num
        self.l = l
        self.u = u

    def forward(self, x):
        x = (x - self.l) / (self.u - self.l) * 2 * pi
        s = [torch.ones_like(x)] + [torch.cos(i * x) for i in range(1, self.basis_num // 2 + 1)] + [torch.sin(i * x) for
                                                                                                    i in range(1,
                                                                                                               self.basis_num // 2 + 1)]
        s = torch.cat(s, 1)
        return s


class FourierBasis1D_decay(Basis):
    def __init__(self, basis_num: int, l: float, u: float):
        assert basis_num % 2 == 1
        super().__init__()
        self.basis_num = basis_num
        self.l = l
        self.u = u

    def forward(self, x):
        x = (x - self.l) / (self.u - self.l) * 2 * pi
        s = [torch.ones_like(x)] + [torch.cos(i * x) / i for i in range(1, self.basis_num // 2 + 1)] + [
            torch.sin(i * x) / i for
            i in range(1,
                       self.basis_num // 2 + 1)]
        s = torch.cat(s, 1)
        return s


class FourierBasis2D(Basis):
    def __init__(self, basis_num: list, l: list, u: list):
        super().__init__()
        """
        example
        basis_num = (5,7)
        l = (-10, 10)
        u = (10, 10)
        """
        self.basis_num = basis_num
        self.l = torch.tensor(l).reshape(2)
        self.u = torch.tensor(u).reshape(2)

    def forward(self, x):
        x = (x - self.l) / (self.u - self.l)
        basis_x = [torch.ones_like(x[:, :1])] + [torch.sin(k * pi * x[:, :1]) / k for k in range(1, self.basis_num[0])]
        basis_y = [torch.ones_like(x[:, 1:])] + [torch.sin(k * pi * x[:, 1:]) / k for k in range(1, self.basis_num[1])]
        s = [b1 * b2 for b1 in basis_x for b2 in basis_y]
        s = torch.cat(s, dim=1)
        return s


class SPHBasis(Basis):
    def __init__(self):
        super().__init__()
        alpha00 = lambda x: torch.ones_like(x)

        alpha10 = lambda x: cos(x)
        alpha11 = lambda x: -sin(x)
        alpha11_ = lambda x: sin(x)

        alpha20 = lambda x: 0.5 * (3 * cos(x) ** 2 - 1)
        alpha21 = lambda x: -3 * sin(x) * cos(x) / 1.5
        alpha22 = lambda x: 3 * sin(x) ** 2 / 3
        alpha22_ = lambda x: 3 * sin(x) ** 2 / 3
        alpha21_ = lambda x: 3 * sin(x) * cos(x) / 1.5

        alpha30 = lambda x: 0.5 * cos(x) * (5 * cos(x) ** 2 - 3)
        alpha31 = lambda x: -1.5 * (5 * cos(x) ** 2 - 1) * sin(x) / 2
        alpha32 = lambda x: 15 * cos(x) * sin(x) ** 2 / 5
        alpha33 = lambda x: -15 * sin(x) ** 3 / 15
        alpha33_ = lambda x: 15 * sin(x) ** 3 / 15
        alpha32_ = lambda x: 15 * cos(x) * sin(x) ** 2 / 5
        alpha31_ = lambda x: 1.5 * (5 * cos(x) ** 2 - 1) * sin(x) / 2

        self.alp_set = {}
        self.alp_set[0] = {0: alpha00}
        self.alp_set[1] = {-1: alpha11_, 0: alpha10, 1: alpha11}
        self.alp_set[2] = {-2: alpha22_, -1: alpha21_, 0: alpha20, 1: alpha21, 2: alpha22}
        self.alp_set[3] = {-3: alpha33_, -2: alpha32_, -1: alpha31_, 0: alpha30, 1: alpha31, 2: alpha32, 3: alpha33}

    def sph(self, x, m=0, l=0):
        assert l >= abs(m)
        if m > 0:
            return torch.cos(m * x[:, 1:]) * self.alp_set[l][m](x[:, :1])
        elif m < 0:
            return torch.sin(-m * x[:, 1:]) * self.alp_set[l][-m](x[:, :1])
        else:
            return self.alp_set[l][0](x[:, :1])

    def forward(self, x):
        sph_basis = torch.cat(
            [self.sph(x, 0, 0), self.sph(x, -1, 1), self.sph(x, 0, 1), self.sph(x, 1, 1), self.sph(x, -2, 2),
             self.sph(x, -1, 2), self.sph(x, 0, 2),
             self.sph(x, 1, 2), self.sph(x, 2, 2), self.sph(x, -3, 3), self.sph(x, -2, 3),
             self.sph(x, -1, 3), self.sph(x, 0, 3),
             self.sph(x, 1, 3), self.sph(x, 2, 3), self.sph(x, 3, 3)], dim=1)
        return sph_basis

    def get_plot(self):
        fig = plt.figure(figsize=(28, 28))
        for l in range(4):
            for m in range(-l, l + 1, 1):
                def net(x):
                    return self.sph(x, m, l)

                # ax = fig.add_subplot(7, 4, 4 * (m + 3) + l + 1, projection='3d')
                ax = fig.add_subplot(7, 4, 4 * (m + 3) + l + 1)
                ax.grid(True)
                self.plot_planner(net, ax)
        plt.show()

    @staticmethod
    def plot_planner(net, ax=None):
        weidu = torch.linspace(0, math.pi, 50, requires_grad=False)
        jingdu = torch.linspace(0, 2 * math.pi, 100, requires_grad=False)
        weidu, jingdu = torch.meshgrid(weidu, jingdu)
        location = torch.cat([weidu.reshape(-1, 1), jingdu.reshape(-1, 1)], dim=1)
        value = net(location)
        value = value.reshape((50, 100))
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 6))
        sns.set()
        value = value.detach().numpy()
        sns.heatmap(value, vmin=value.min(), vmax=value.max(), ax=ax)
        ax.set_xticks([])
        ax.set_yticks([])


if __name__ == '__main__':
    SPHBasis().get_plot()
