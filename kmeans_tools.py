import torch
from rules import RuleBase
from dataset import Dataset
from typing import List


class KmeansUtils(object):
    def __init__(self):
        useless = 0

    def kmeans_admm(self, para_rho, data: List[Dataset], n_rules, n_agents, rules: RuleBase):
        # rules = self.rules_kmeans(data.X, n_rules)
        # calculate H matrix and w for each node
        # parameters initialize
        rho = para_rho
        max_steps = 300
        admm_reltol = 0.001
        admm_abstol = 0.0001

        errors = []
        # initiate lagrange multiplier
        lagrange_mul = torch.zeros(n_agents, n_rules, data[1].X.shape[1]).double()
        # initiate global term centrio mu
        rules.fit(data[1].X, n_rules)
        center_global = rules.center_list
        # initiate global center set
        center_agent_set = torch.zeros((n_agents, center_global.shape[0], center_global.shape[1])).double()

        for i in torch.arange(max_steps):

            for j in torch.arange(n_agents):
                # assign clusters for each node based on the global centoids
                # ye's method
                # dist_x = center_global.centerl(center_global) / 2 - center_global.mm(data[j].X.t())
                # labels = torch.min(dist_x, 0)[1]
                rules.update_rules(data[int(j)].X, center_global)

                center_agent_set[j, :, :] = rules.update_center(data[int(j)].X)

            # store the old global centrio and update
            center_global_old = center_global.clone()

            # for each cluster
            for j in torch.arange(n_rules):
                center_global[j, :] = (rho * center_agent_set[:, j, :].sum(0)
                                       + lagrange_mul[:, j, :].sum(0)) / (rho * n_agents)

            # compute the update for the Lagrange Mltipliers
            for j in torch.arange(n_rules):
                for k in torch.arange(n_agents):
                    lagrange_mul[k, j, :] = lagrange_mul[k, j, :] +\
                        rho * (center_agent_set[k, j, :] - center_global[j, :])
            # check stoping criterion
            stop_crtn = - rho * (center_global - center_global_old)

            errors.append(torch.norm(stop_crtn))

            # if errors[int(i)] < torch.sqrt(torch.tensor(n_agents).double()) * admm_abstol:
            #     break
        center_optimal = center_global

        return center_optimal, torch.tensor(errors)
