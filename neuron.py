import abc
import torch
from dataset import Dataset
from fnn_utils import fnn_admm
from h_utils import HBase, HNormal
from kmeans_tools import KmeansUtils
from rules import RuleBase, RuleKmeans
from fnn_solver import FnnSolveBase, FnnSolveReg
from partition import KFoldPartition


class Neuron(object):
    """
    the base class of the fuzzy neuron
    """
    def __init__(self, rules_seed: RuleBase, h_computer_seed: HBase,
                 fnn_solver_seed: FnnSolveBase):
        self.__rules = type(rules_seed)()
        self.__h_computer = type(h_computer_seed)()
        self.__fnn_solver = type(fnn_solver_seed)()

    def predict(self, data: Dataset):
        # update rules on test data
        self.__rules.update_rules(data.X, self.__rules.center_list)
        h, _ = self.__h_computer.comute_h(data.X, self.__rules)
        n_rule = h.shape[0]
        n_smpl = h.shape[1]
        n_fea = h.shape[2]
        h_cal = h.permute((1, 0, 2))  # N * n_rules * (d + 1)
        h_cal = h_cal.reshape(n_smpl, n_rule * n_fea)  # squess the last dimension

        # calculate Y hat
        y_hat = h_cal.mm(self.__rules.consequent_list.reshape(data.Y.shape[1],
                                                              n_rule * n_fea).t())
        return y_hat

    def clone(self):
        new_neuron = type(self)(type(self.get_rules())(),
                                type(self.get_h_computer())(),
                                type(self.get_fnn_solver())())
        return new_neuron

    @abc.abstractmethod
    def forward(self, **kwargs):
        pass

    def set_rules(self, rules: RuleBase):
        self.__rules = rules

    def get_rules(self):
        return self.__rules

    def set_h_computer(self, h_computer: HBase):
        self.__h_computer = h_computer

    def get_h_computer(self):
        return self.__h_computer

    def set_fnn_solver(self, fnn_solver: FnnSolveBase):
        self.__fnn_solver = fnn_solver

    def get_fnn_solver(self):
        return self.__fnn_solver

    def clear(self):
        self.set_rules(type(self.get_rules())())
        self.set_h_computer(type(self.get_h_computer())())
        self.set_fnn_solver(type(self.get_fnn_solver())())


class NeuronC(Neuron):
    def __init__(self, rules_seed: RuleBase, h_computer_seed: HBase,
                 fnn_solver_seed: FnnSolveBase):
        super(NeuronC, self).__init__(rules_seed, h_computer_seed,
                                      fnn_solver_seed)

    def forward(self, **kwargs):
        data: Dataset = kwargs['data']
        para_mu = kwargs['para_mu']
        n_rules = kwargs['n_rules']
        rules = self.get_rules()
        h_computer = self.get_h_computer()
        fnn_solver = self.get_fnn_solver()
        rules.fit(data.X, n_rules)
        h, _ = h_computer.comute_h(data.X, rules)
        # run FNN solver for given rule number
        fnn_solver.h = h
        fnn_solver.y = data.Y.double()
        fnn_solver.para_mu = para_mu
        w_optimal = fnn_solver.solve()
        rules.consequent_list = w_optimal
        self.set_rules(rules)


class NeuronD(Neuron):
    def __init__(self, rules_seed: RuleBase, h_computer_seed: HBase,
                 fnn_solver_seed: FnnSolveBase):
        super(NeuronD, self).__init__(rules_seed, h_computer_seed,
                                      fnn_solver_seed)

    def forward(self, **kwargs):
        # get parameters from kwarg
        data: Dataset = kwargs['data']
        para_mu = kwargs['para_mu']
        para_rho = kwargs['para_rho']
        n_agents = kwargs['n_agents']
        n_rules = kwargs['n_rules']

        d_data = data.get_subset_smpl(KFoldPartition(n_agents))

        rules = self.get_rules()
        h_computer = self.get_h_computer()
        fnn_solver = self.get_fnn_solver()
        # train distributed fnn
        kmeans_utils = KmeansUtils()
        center_optimal, errors = kmeans_utils.kmeans_admm(para_rho, d_data, n_rules,
                                                          n_agents, rules)
        n_fea = d_data[1].X.shape[1]
        n_output = d_data[1].Y.shape[1]
        h_all_agent = []
        # the shape of w set is n_agents *  n_output * n_rules * len_w
        w_all_agent = torch.empty((0, n_output,
                                   n_rules, n_fea + 1)).double()

        for i in torch.arange(n_agents):
            rules.update_rules(d_data[int(i)].X, center_optimal)
            h_per_agent, _ = h_computer.comute_h(d_data[int(i)].X, rules)
            h_all_agent.append(h_per_agent)

            fnn_solver.h = h_per_agent
            fnn_solver.y = d_data[int(i)].Y.double()
            fnn_solver.para_mu = para_mu
            w_optimal_per_agent = fnn_solver.solve()
            w_all_agent = torch.cat((w_all_agent, w_optimal_per_agent.unsqueeze(0)), 0)

        w_optimal_all_agent, z, errors = fnn_admm(d_data, para_mu, n_agents, n_rules,
                                                  w_all_agent, h_all_agent)

        rules.consequent_list = z
        self.set_rules(rules)


class NeuronDC(Neuron):
    """a FNN structure with distributed antecedent and centralized consequent layer"""
    def __init__(self, rules_seed: RuleBase, h_computer_seed: HBase,
                 fnn_solver_seed: FnnSolveBase):
        super(NeuronDC, self).__init__(rules_seed, h_computer_seed,
                                       fnn_solver_seed)

    def forward(self, **kwargs):
        # get parameters from kwarg
        data: Dataset = kwargs['data']
        para_mu = kwargs['para_mu']
        para_rho = kwargs['para_rho']
        n_agents = kwargs['n_agents']
        n_rules = kwargs['n_rules']

        d_data = data.get_subset_smpl(KFoldPartition(n_agents))

        rules = self.get_rules()
        h_computer = self.get_h_computer()
        fnn_solver = self.get_fnn_solver()
        # train distributed fnn
        kmeans_utils = KmeansUtils()
        center_optimal, _ = kmeans_utils.kmeans_admm(para_rho, d_data, n_rules,
                                                     n_agents, rules)
        rules.update_rules(data.X, center_optimal)
        h, _ = h_computer.comute_h(data.X, rules)
        # run FNN solver for given rule number
        fnn_solver.h, _ = h
        fnn_solver.y = data.Y.double()
        fnn_solver.para_mu = para_mu
        w_optimal = fnn_solver.solve()
        rules.consequent_list = w_optimal
        self.set_rules(rules)


class NeuronCD(Neuron):
    """a FNN structure with centralized antecedent and distributed consequent layer"""
    def __init__(self, rules_seed: RuleBase, h_computer_seed: HBase,
                 fnn_solver_seed: FnnSolveBase):
        super(NeuronCD, self).__init__(rules_seed, h_computer_seed,
                                      fnn_solver_seed)

    def forward(self, **kwargs):
        # get parameters from kwarg
        data: Dataset = kwargs['data']
        para_mu = kwargs['para_mu']
        para_rho = kwargs['para_rho']
        n_agents = kwargs['n_agents']
        n_rules = kwargs['n_rules']

        d_data = data.get_subset_smpl(KFoldPartition(n_agents))

        rules = self.get_rules()
        h_computer = self.get_h_computer()
        fnn_solver = self.get_fnn_solver()
        # train distributed fnn
        rules.fit(data.X, n_rules)
        center_optimal = rules.center_list
        n_fea = d_data[1].X.shape[1]
        n_output = d_data[1].Y.shape[1]
        h_all_agent = []
        # the shape of w set is n_agents *  n_output * n_rules * len_w
        w_all_agent = torch.empty((0, n_output,
                                   n_rules, n_fea + 1)).double()

        for i in torch.arange(n_agents):
            rules.update_rules(d_data[int(i)].X, center_optimal)
            h_per_agent = h_computer.comute_h(d_data[int(i)].X, rules)
            h_all_agent.append(h_per_agent)

            fnn_solver.h, _ = h_per_agent
            fnn_solver.y = d_data[int(i)].Y.double()
            fnn_solver.para_mu = para_mu
            w_optimal_per_agent = fnn_solver.solve()
            w_all_agent = torch.cat((w_all_agent, w_optimal_per_agent.unsqueeze(0)), 0)

        w_optimal_all_agent, z, errors = fnn_admm(d_data, para_mu, n_agents, n_rules,
                                                  w_all_agent, h_all_agent)

        rules.consequent_list = z
        self.set_rules(rules)


class NeuronDN(Neuron):
    """a FNN structure with distributed antecedent and centralized consequent layer"""
    def __init__(self, rules_seed: RuleBase, h_computer_seed: HBase,
                 fnn_solver_seed: FnnSolveBase):
        super(NeuronDN, self).__init__(rules_seed, h_computer_seed,
                                       fnn_solver_seed)

    def forward(self, **kwargs):
        # get parameters from kwarg
        data: Dataset = kwargs['data']
        para_rho = kwargs['para_rho']
        n_agents = kwargs['n_agents']
        n_rules = kwargs['n_rules']

        d_data = data.get_subset_smpl(KFoldPartition(n_agents))

        rules = self.get_rules()
        h_computer = self.get_h_computer()
        # train distributed fnn
        kmeans_utils = KmeansUtils()
        center_optimal, _ = kmeans_utils.kmeans_admm(para_rho, d_data, n_rules,
                                                     n_agents, rules)
        rules.update_rules(data.X, center_optimal)
        _, w = h_computer.comute_h(data.X, rules)
        # run FNN solver for given rule number

        rules.consequent_list = w
        self.set_rules(rules)

    def predict(self, data: Dataset):
        rules = self.get_rules()
        h_computer = self.get_h_computer()
        rules.update_rules(data.X, rules.center_list)
        _, w = h_computer.comute_h(data.X, rules)
        return w
