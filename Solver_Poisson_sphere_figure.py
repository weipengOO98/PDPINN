import torch

from Basis import SPHBasis
from Problem import Problem_Sphere_Poisson
from Model import Model
from Net import Sphere_Net, SPH_Sphere_Net
import seaborn as sns
import matplotlib.pyplot as plt
import math

torch.manual_seed(0)
maxiter = 2001
problem = Problem_Sphere_Poisson()


def construct_model(net):
    class Poisson1dModel(Model):
        def add_loss_history(self):
            self.loss_history.append([self.bc_loss, self.pde_loss, self.predict_error_value])

        def __init__(self):
            super().__init__(problem=problem, net=net, maxiter=maxiter)

        def inner_sample(self, num=200):
            x = torch.randn(num, 3)
            x = x / (torch.norm(x, dim=1).reshape((-1, 1)))
            weidu = torch.acos(x[:, 2:3])
            jingdu = torch.atan(x[:, :1] / x[:, 1:2]) + math.pi / 2
            jingdu[:num // 2, 0] = jingdu[:num // 2, 0] + math.pi
            x = torch.cat([weidu, jingdu], dim=1)
            return x

        def bc_sample(self):
            return torch.tensor([[1., 1.]])

        def init_sample(self):
            pass

        def plot(self, net, ax):
            weidu = torch.linspace(0, math.pi, 200, requires_grad=False)
            jingdu = torch.linspace(0, 2 * math.pi, 400, requires_grad=False)
            weidu, jingdu = torch.meshgrid(weidu, jingdu)
            location = torch.cat([weidu.reshape(-1, 1), jingdu.reshape(-1, 1)], dim=1)
            value = net(location)
            value = value.reshape((200, 400))
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(8, 6))
            sns.set()
            sns.heatmap(value.detach().numpy(), ax=ax, vmin=-0.5, vmax=0.5, cbar=False)
            ax.set_xticks([])
            ax.set_yticks([])

        def post_process(self, ax=None):
            if ax is None:
                plt.plot(self.loss_history)
                plt.yscale('log')
                plt.legend(('BC loss', 'pde loss', 'predict error'))
            else:
                ax.plot(self.loss_history)
                ax.set_yscale('log')
                ax.set_ylim(1e-4, 100)
                ax.legend(('BC loss', 'pde loss', 'predict error'))

        def predict_error(self):
            coor = self.inner_sample(num=1000)
            true = self.problem.ground_truth(coor)
            predict = self.net(coor)
            predict_error = self.pde_loss_f(true, predict)
            return predict_error

        def train(self, ax100=None):
            problem = self.problem
            net = self.net
            opt = self.opt
            maxiter = self.maxiter

            for iter in range(maxiter):
                net.zero_grad()

                coor_inner = self.inner_sample().detach().requires_grad_(True)
                infer_value_inner = net(coor_inner)
                truth_inner, predict_inner = problem.pde(coor_inner, infer_value_inner)
                self.pde_loss = self.pde_loss_f(predict_inner, truth_inner)

                bc_samples = self.bc_sample()
                if bc_samples is None:
                    self.bc_loss = torch.tensor(0.)
                else:
                    coor_bc = bc_samples.detach().requires_grad_(True)
                    infer_value_bc = net(coor_bc)
                    truth_bc, predict_bc = problem.bound_condition(coor_bc, infer_value_bc)
                    self.bc_loss = self.bc_loss_f(predict_bc, truth_bc)

                init_samples = self.init_sample()
                if init_samples is None:
                    self.init_loss = torch.tensor(0.)
                else:
                    coor_init = init_samples.detach().requires_grad_(True)
                    infer_value_init = net(coor_init)
                    truth_init, predict_init = problem.bound_condition(coor_init, infer_value_init)
                    self.init_loss = self.bc_loss_f(predict_init, truth_init)
                self.predict_error_value = self.predict_error()
                self.total_loss = self.pde_loss + self.bc_loss + self.init_loss
                self.add_loss_history()
                self.total_loss.backward()
                opt.step()
                opt.zero_grad()
                if iter % (maxiter // 100) == 0:
                    print("iteration {}: loss = {}".format(iter, self.total_loss))
                if iter == 100:
                    if ax100: self.plot(self.net, ax100)

    return Poisson1dModel()


fig2 = plt.figure(constrained_layout=False, figsize=(16, 10))
grid = fig2.add_gridspec(2, 3)
ax = [[None, None], [None, None]]

ax[0][0] = fig2.add_subplot(grid[0, :2])
ax[0][1] = fig2.add_subplot(grid[0, 2])
ax[1][0] = fig2.add_subplot(grid[1, :2])
ax[1][1] = fig2.add_subplot(grid[1, 2])

basis = SPHBasis()
net1 = SPH_Sphere_Net([3, 50, 50, 50, 16], basis)
net0 = Sphere_Net([3, 50, 50, 50, 16, 1])

net_table = [net0, net1]
for i, net in enumerate(net_table):
    print('{}-th net'.format(i))
    model = construct_model(net)
    model.train()
    model.plot(model.net, ax[i][0])
    model.post_process(ax[i][1])

ax[0][0].set_ylabel('Sphere-PINN')
ax[1][0].set_ylabel('Sphere-PINN-PD')
ax[1][0].set_xlabel('Iteration 2000')
ax[1][1].set_xlabel('Loss')
plt.show()
