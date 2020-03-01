import torch
from dataset import Dataset
from rules import RuleBase
from typing import List
from loss_utils import LossFunc
from h_utils import HBase


def fnn_centralized(self, x, y, max_n_rules, loss_functions: LossFunc, h_compute: HBase):
    # try all possible rule methods
    rules_set: List[RuleBase] = []
    h_set = []
    y_hat_set = []
    w_optimal_tmp_set = []
    loss_set = torch.zeros(max_n_rules)
    for i in torch.arange(max_n_rules):
        ruls_tmp = RuleBase()
        ruls_tmp.fit(x, i)
        rules_set.append(ruls_tmp)
        h_tmp = h_compute.comute_h(x, ruls_tmp)
        h_set.append(h_tmp)

        # run FNN solver for all each rule number
        w_optimal_tmp_tmp, y_hat_tmp = self.fnn_solve_r(h_tmp, y)
        w_optimal_tmp_set.append(w_optimal_tmp_tmp)
        y_hat_set.append(y_hat_tmp)
        loss_set[i] = loss_functions.forward(y_hat_tmp, y)

    min_loss = loss_set.min(0)
    min_idx = int(min_loss[1])
    n_rules = min_idx
    rules = rules_set[min_idx]
    y_hat = y_hat_set[min_idx]
    h_best = h_set[min_idx]
    w_optimal_tmp = w_optimal_tmp_set[min_idx]
    # w is a column vector
    for i in torch.arange(n_rules):
        rules_set[int(i)].consequent_list = w_optimal_tmp[i, :]

    return y_hat, min_loss, loss_set, rules, n_rules, h_best, w_optimal_tmp


def fnn_admm(d_train_data: List[Dataset], param_mu, n_agents, n_rules, w, h):
    # parameters initialize
    rho = 1
    max_steps = 300
    admm_reltol = 0.001
    admm_abstol = 0.001

    n_node = n_agents
    n_fea = d_train_data[1].X.shape[1]
    n_output = d_train_data[1].Y.shape[1]

    len_w = n_rules * (n_fea + 1)

    errors = torch.zeros(max_steps, n_node).double()

    z = torch.zeros(n_output, len_w).double()
    lagrange_mul = torch.zeros(n_node, n_output, len_w).double()

    # precompute the matrices
    h_inv = torch.zeros(n_node, len_w, len_w).double()
    h_y = torch.zeros(n_node, n_output, len_w).double()

    for i in torch.arange(n_node):
        h_tmp = h[i]
        n_smpl = h_tmp.shape[1]
        h_cal = h_tmp.permute((1, 0, 2))  # N * n_rules * (d + 1)
        h_cal = h_cal.reshape(n_smpl, len_w)
        h_inv[i, :, :] = torch.inverse(torch.eye(len_w).double() * rho + h_cal.t().mm(h_cal))
        h_y[i, :, :] = h_cal.t().mm(d_train_data[int(i)].Y.double()).t()

    w_cal = []
    for i in torch.arange(max_steps):
        for j in torch.arange(n_node):
            w_cal = w.reshape(n_node, n_output, len_w)
            w_cal[j, :, :] = h_inv[j, :, :].mm((h_y[j, :, :] +
                                                rho * z - lagrange_mul[j, :, :]).t()).t()

        # store the old z while update it
        z_old = z.clone()
        z = (rho * torch.sum(w_cal, 0) + torch.sum(lagrange_mul, 0)) / (param_mu + rho * n_node)

        # compute the update for the lagrangian multipliers
        for j in torch.arange(n_node):
            lagrange_mul[j, :, :] = lagrange_mul[j, :, :] + rho * (w_cal[j, :, :] - z)

        # check stopping criterion
        z_norm = rho * (z - z_old)
        lagrange_mul_norm = torch.zeros(n_node).double()

        primal_criterion = torch.zeros(n_node).double()

        for j in torch.arange(n_node):
            w_error = w_cal[j, :] - z
            errors[i, j] = torch.norm(w_error)
            if errors[i, j] < torch.sqrt(torch.tensor(n_node).double()) * admm_abstol + \
                    admm_reltol * torch.max(torch.norm(w[j, :], 2), torch.norm(z, 2)):
                primal_criterion[j] = 1
            lagrange_mul_norm[j] = torch.norm(lagrange_mul[j, :, :], 2)

        if torch.norm(z_norm) < torch.sqrt(torch.tensor(n_node).double()) * admm_abstol + admm_reltol * \
                lagrange_mul_norm.max() and primal_criterion.max() == 1:
            break
    # w_cal = w_cal.reshape(n_node, n_rules, (n_fea + 1))
    return w_cal, z.reshape(n_output, n_rules, (n_fea + 1)), errors






