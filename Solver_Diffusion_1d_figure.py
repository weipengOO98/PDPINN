import torch

from Basis import FourierBasis1D
from Net import Net, Basis_Net_Time
from Problem import Problem_1d_diffussion
from Model import Model
import matplotlib.pyplot as plt
import seaborn as sns

torch.manual_seed(0)
problem = Problem_1d_diffussion()
maxiter = 1001


def construct_model(net):
    class Poisson1dModel(Model):
        def add_loss_history(self):
            self.loss_history.append([self.bc_loss, self.pde_loss, self.predict_error_value, self.init_loss])

        def __init__(self):
            super().__init__(problem=problem, net=net, maxiter=maxiter)

        def inner_sample(self, num=1000):
            num = 1000
            return torch.rand((num, 2)).cpu() * torch.tensor([20., 1.]).cpu() - torch.tensor([10., 0]).cpu()

        def bc_sample(self):
            num = 100
            a = torch.cat([torch.ones((num, 1)).cpu() * (-10.), torch.rand((num, 1)).cpu()], dim=1)
            b = torch.cat([torch.ones((num, 1)).cpu() * 10, torch.rand((num, 1)).cpu()], dim=1)
            return torch.cat([a, b], 0)

        def init_sample(self):
            num = 100
            return torch.cat([torch.rand((num, 1)).cpu() * 20 - 10, torch.zeros((num, 1)).cpu()], dim=1)

        def plot(self, net, ax):
            xx = torch.linspace(-10, 10, 400).cpu()
            tt = torch.linspace(0, 1, 400).cpu()
            x1, y1 = torch.meshgrid([xx, tt])
            s1 = x1.shape
            x1 = x1.reshape((-1, 1))
            y1 = y1.reshape((-1, 1))
            x = torch.cat([x1, y1], dim=1)
            z = net(x).reshape(s1)
            sns.set()
            sns.heatmap(z.cpu().detach().numpy(), vmin=-3, vmax=3, ax=ax)
            ax.set_xticks([])
            ax.set_yticks([])

        def post_process(self, ax=None):
            if ax is None:
                plt.plot(self.loss_history)
                plt.yscale('log')
                plt.legend(('BC loss', 'pde loss', 'predict error', 'IC loss'))
            else:
                ax.plot(self.loss_history)
                ax.set_yscale('log')
                ax.set_ylim(1e-3, 10)
                ax.legend(('BC loss', 'pde loss', 'predict error', 'IC loss'))

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
            # _, axe = plt.subplots(1, 10, figsize=(50, 5))
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

basis = FourierBasis1D(21, -10, 10)
net1 = Basis_Net_Time([2, 50, 50, 50, 21], basis=basis)
net0 = Net([2, 50, 50, 50, 50, 1])

net_table = [net0, net1]
for i, net in enumerate(net_table):
    print('{}-th net'.format(i))
    model = construct_model(net)
    model.train()
    model.plot(model.net, ax[i][0])
    model.post_process(ax[i][1])

ax[0][0].set_ylabel('PINN')
ax[1][0].set_ylabel('PINN-PD(k=10)')
ax[1][0].set_xlabel('Iteration 1000')
ax[1][1].set_xlabel('Loss')
plt.show()
