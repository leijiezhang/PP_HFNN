import torch
from cvxopt import matrix, solvers
from sklearn import kernel_approximation


x_j = torch.rand(1000, 12)
y_j = torch.rand(1000, 1)
n_agents = 5
n_fea = x_j.shape[1]
n_smpl = x_j.shape[1]
x_j = torch.cat((x_j, torch.ones(n_smpl).unsqueeze(1)), 0)

eta = 0.01
c = 0.1

i_mtx = torch.eye(n_fea + 1)
pie_mtx = torch.zeros(n_fea + 1, n_fea + 1)
pie_mtx[n_fea, n_fea] = 1

v = torch.zeros(n_agents, n_fea + 1)
alpha_mtx = torch.zeros(n_agents, n_fea + 1)
lambda_mtx = torch.zeros(n_agents, n_fea + 1)


u = (1 + 2 * eta) * i_mtx - pie_mtx
for j in torch.arange(n_agents):
    v_j_sum = (eta * v[j, :].repeat(n_agents, 1) + v).sum(0)
    f_j = 2 * alpha_mtx[j, :] - v_j_sum
    lambda_j_p = y_j.mm(x_j).mm(torch.inverse(u)).mm(x_j.t()).mm(y_j)
    # lambda_j_p = matrix(lambda_j_p)
    lambda_j_q = torch.ones(n_smpl) + y_j.mm(x_j).mm(torch.inverse(u)).mm(f_j.unsqueeze(1))
    lambda_j_g = torch.cat((torch.eye(n_smpl), -1 * torch.eye(n_smpl)), 0)
    lambda_j_h = torch.cat((n_agents * c * torch.ones(n_smpl), torch.zeros(n_smpl)), 0)
    lambda_j_a = torch.zeros(n_smpl, n_smpl)
    lambda_j_b = torch.zeros(n_smpl, 1)

    lambda_mtx[j, :] = solvers.qp(lambda_j_p, lambda_j_q, lambda_j_g, lambda_j_h, lambda_j_a, lambda_j_b)

    v[j, :] = torch.inverse(u).mm(x_j.t().mm(y_j).mm(lambda_mtx[j, :]) - f_j)
    v_j_cut = (v[j, :].repeat(n_agents, 1) - v).sum(0)
    alpha_mtx[j, :] = alpha_mtx[j, :] + (eta / 2) * v_j_cut


def kernel_rbf(x: torch.Tensor, x_t = None):
    """
    K(x, x')
    :param x: 
    :return: 
    """
    rbf_feature = kernel_approximation.RBFSampler(gamma=1, random_state=1)
    x_kenel = rbf_feature.fit_transform(x)
    if x_t is None:
        x_t = x_kenel.t()
    else:
        x_t = x_t.t()
    return x_kenel.mm(x_t)


# non-linear distributed svm
x_kernel = kernel_rbf(x_j)
l = x_kernel.shape[0]
x_all = torch.rand(2000, 12)
tao = x_all[0:l, :]

w_hat = torch.ones(n_agents, l)
f_hat = torch.ones(n_agents, l)
h = torch.ones(n_agents)
a = torch.zeros(n_agents, n_smpl)
b = torch.ones(n_agents)
c = torch.zeros(n_agents, l)

lambda_mtx = torch.zeros(n_agents, n_smpl)
alpha_mtx = torch.zeros(n_agents, n_smpl)
beta_arr = torch.zeros(n_agents)


def kernel_hat_rbf(x_j, x_j_t=None):
    if x_j_t is None:
        x_j_t = x_j
    kernel_hat = 2 * eta * n_agents * kernel_rbf(x_j, tao).mm(torch.inverse(u_hat)).mm(kernel_rbf(tao, x_j_t))
    return kernel_hat

for j in torch.arange(n_agents):
    u_hat = torch.eye(l) + 2 * eta * n_agents * kernel_rbf(tao)
    w_hat_j_sum = (w_hat[j, :].repeat(n_agents, 1) + w_hat).sum(0)
    f_hat[j, :] = 2 * alpha_mtx[j, :] - eta * w_hat_j_sum
    h[j] = 2 * beta_arr[j] - eta * (n_agents * b[j] + b.sum(0))


    lambda_j_p = y_j.mm(kernel_rbf(x_j)) + kernel_hat_rbf(x_j) + 1/(2 * eta * n_agents)
    # lambda_j_p = matrix(lambda_j_p)
    lambda_j_q = f_hat[j, :].mm(kernel_rbf(tao, x_j) - kernel_hat_rbf(tao, x_j)) + h[j]/(2 * eta * n_agents) * torch.ones(n_smpl)

    lambda_j_g = torch.cat((torch.eye(n_smpl), -1 * torch.eye(n_smpl)), 0)
    lambda_j_h = torch.cat((n_agents * c * torch.ones(n_smpl), torch.zeros(n_smpl)), 0)
    lambda_j_a = torch.zeros(n_smpl, n_smpl)
    lambda_j_b = torch.zeros(n_smpl, 1)

    w_hat[j, :] = (kernel_rbf(tao, x_j) - kernel_hat_rbf(tao, x_j)).mm(y_j).mm(alpha_mtx[j, :]) - (kernel_rbf(tao) - kernel_hat_rbf(tao)).mm(f_hat[j, :])
    b[j] = 1/(2 * eta * n_agents) * (torch.ones(1, n_smpl).mm(y_j).mm(alpha_mtx[j, :]) - h[j])

    w_hat_j_min = (w_hat[j, :].repeat(n_agents, 1) - w_hat).sum(0)
    alpha_mtx[j, :] = alpha_mtx[j, :] + eta/2 * w_hat_j_min
    beta_arr[j] = beta_arr[j] + eta/2 * (beta_arr[j] * n_agents - beta_arr.sum(0))

    a[j, :] = y_j.mm(alpha_mtx[j, :])
    c[j, :] = 2*eta*n_agents * torch.inverse(u_hat).mm(kernel_rbf(tao).mm(f_hat[j, :]) - kernel_rbf(tao, x_j).mm(y_j).mm(alpha_mtx[j, :])) - f_hat[j, :]
    b[j] = 1/(2 * eta * n_agents) * (torch.ones(1, n_smpl).mm(y_j).mm(alpha_mtx[j, :]) - h[j])
