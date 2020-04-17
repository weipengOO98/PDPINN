import torch

from Basis import FourierBasis2D
from Problem import Problem_2d_Poisson
from Model import Model
from Net import Net, Basis_Net
import seaborn as sns
import matplotlib.pyplot as plt

torch.manual_seed(0)
problem = Problem_2d_Poisson()
maxiter = 1001


def construct_model(net):
    class Poisson1dModel(Model):
        def add_loss_history(self):
            self.loss_history.append([self.bc_loss, self.pde_loss, self.predict_error_value])

        def __init__(self):
            super().__init__(problem=problem, net=net, maxiter=maxiter)

        def inner_sample(self, num=1000):
            return torch.rand(num, 2).cpu() * 20 - 10

        def bc_sample(self):
            num = 400
            x = torch.rand(num, 2).cpu() * 20 - 10
            num = num // 4
            x[:num, 1] = 10.
            x[num + 1:2 * num, 0] = 10
            x[2 * num + 1:3 * num, 1] = -10
            x[3 * num + 1:4 * num, 0] = -10
            return x

        def init_sample(self):
            pass

        def plot(self, net, ax):
            xx = torch.linspace(-10, 10, 400).cpu()
            yy = torch.linspace(-10, 10, 400).cpu()
            x1, y1 = torch.meshgrid([xx, yy])
            s1 = x1.shape
            x1 = x1.reshape((-1, 1))
            y1 = y1.reshape((-1, 1))
            x = torch.cat([x1, y1], dim=1)
            z = net(x).reshape(s1)
            sns.set()
            sns.heatmap(z.cpu().detach().numpy().T, vmin=-4, vmax=4, ax=ax, cbar=False)
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
                ax.set_ylim(1e-4, 10)
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
                    self.plot(self.net, ax100)

    return Poisson1dModel()


fig, ax = plt.subplots(2, 3, figsize=(16, 10))

basis = FourierBasis2D([5, 5], [-10., -10.], [10., 10.])
net1 = Basis_Net([2, 50, 50, 50, 25], basis)
net0 = Net([2, 50, 50, 50, 25, 1])

net_table = [net0, net1]
for i, net in enumerate(net_table):
    print('{}-th net'.format(i))
    model = construct_model(net)
    model.train(ax[i][0])
    model.plot(model.net, ax[i][1])
    model.post_process(ax[i][2])
ax[0][0].set_ylabel('PINN(hidden-layer=3)')
ax[1][0].set_ylabel('PINN-PD(k1=k2=5)')
ax[1][0].set_xlabel('Iteration 100')
ax[1][1].set_xlabel('Iteration 1000')
ax[1][2].set_xlabel('Loss')
plt.show()
