import torch
from Problem import Problem_1d_bodong
from Basis import FourierBasis1D
from Net import Basis_Net, Net
from Model import Model
import matplotlib.pyplot as plt

torch.manual_seed(0)
problem = Problem_1d_bodong()
maxiter = 1001


def construct_model(net):
    class Poisson1dModel(Model):
        def add_loss_history(self):
            self.loss_history.append([self.bc_loss, self.pde_loss, self.predict_error_value])

        def __init__(self):
            super().__init__(problem=problem, net=net, maxiter=maxiter)

        def inner_sample(self, num=100):
            sample_num = num
            return torch.rand((sample_num, 1)) * 20. - 10.

        def bc_sample(self):
            return torch.tensor([[-10.], [10.]])

        def init_sample(self):
            pass

        def plot(self, net, ax):
            xx = torch.linspace(-10, 10, 1000).reshape((-1, 1))
            yy = net(xx)
            zz = self.problem.ground_truth(xx)
            xx = xx.reshape((-1)).data.numpy()
            yy = yy.reshape((-1)).data.numpy()
            zz = zz.reshape((-1)).data.numpy()
            ax.plot(xx, yy)
            ax.plot(xx, zz, 'r')

        def post_process(self, ax=None):
            if ax is None:
                plt.plot(self.loss_history)
                plt.yscale('log')
                plt.legend(('BC loss', 'pde loss', 'predict error'))
            else:
                ax.plot(self.loss_history)
                ax.set_yscale('log')
                ax.set_ylim(1e-8, 10)
                ax.legend(('BC loss', 'pde loss', 'predict error'))

        def predict_error(self):
            coor = self.inner_sample(num=1000)
            true = self.problem.ground_truth(coor)
            predict = self.net(coor)
            predict_error = self.pde_loss_f(true, predict)
            return predict_error

        def train(self):
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

    return Poisson1dModel()


fig, ax = plt.subplots(4, 3, figsize=(16, 16))

basis1 = FourierBasis1D(9, l=-10., u=10.)
net1 = Basis_Net([1, 50, 50, 50, 9], basis1)

basis2 = FourierBasis1D(17, l=-10., u=10.)
net2 = Basis_Net([1, 50, 50, 50, 17], basis2)

net3 = Net([1, 50, 50, 50, 1])

net4 = Net([1, 50, 50, 50, 17, 1])

net_table = [net3, net4, net1, net2]
for i, net in enumerate(net_table):
    print('{}-th net'.format(i))
    model = construct_model(net)
    model.plot(model.net, ax[i][0])
    model.train()
    model.plot(model.net, ax[i][1])
    model.post_process(ax[i][2])
ax[0][0].set_ylabel('PINN(hidden-layer=3)')
ax[1][0].set_ylabel('PINN(hidden-layer=4)')
ax[2][0].set_ylabel('PINN-PD(k=4)')
ax[3][0].set_ylabel('PINN-PD(k=8)')
ax[3][0].set_xlabel('Iteration 0')
ax[3][1].set_xlabel('Iteration 1000')
ax[3][2].set_xlabel('Loss')
plt.show()
