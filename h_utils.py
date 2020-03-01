import torch
import skfuzzy
from rules import RuleBase, RuleKmeans, RuleFuzzyCmeans
import abc


class HBase(object):
    def __init__(self):
        pass

    @abc.abstractmethod
    def comute_h(self, x: torch.Tensor, rules: RuleBase):
        h = []
        return h


class HNormal(HBase):
    def __init__(self):
        super(HNormal, self).__init__()

    def comute_h(self, x: torch.Tensor, rules: RuleBase):
        n_smpl = x.shape[0]
        n_fea = x.shape[1]
        n_rules = rules.n_rules
        mf_set = torch.zeros(n_rules, n_smpl, n_fea)
        for i in torch.arange(n_rules):
            for j in torch.arange(n_fea):
                mf = skfuzzy.membership.gaussmf(x[:, j], rules.center_list[i][j],
                                                rules.widths_list[i][j])
                mf_set[i, :, j] = mf

        w = torch.prod(mf_set.double(), 2)
        w_hat = w / torch.sum(w, 0).repeat(n_rules, 1)
        w_hat[torch.isnan(w_hat)] = 1 / n_rules

        h = torch.empty(0, n_smpl, n_fea + 1).double()
        for i in torch.arange(n_rules):
            w_hat_per_rule = w_hat[i, :].unsqueeze(1).repeat(1, n_fea + 1)
            x_extra = torch.cat((torch.ones(n_smpl, 1).double(), x), 1)
            h_per_rule = torch.mul(w_hat_per_rule, x_extra).unsqueeze(0)
            h = torch.cat((h, h_per_rule), 0)
        return h, w_hat.t()


class HFuzzy(HBase):
    def __init__(self):
        super(HFuzzy, self).__init__()

    def comute_h(self, x: torch.Tensor, rules: RuleFuzzyCmeans):
        """
            todo: using fuzzy partition instead of fuzzy sets to get h
            :return:
            """
        n_smpl = x.shape[0]
        n_fea = x.shape[1]
        n_rules = rules.n_rules

        w_hat = rules.data_partition.t()
        w_hat[torch.isnan(w_hat)] = 1 / n_rules

        h = torch.empty(0, n_smpl, n_fea + 1).double()
        for i in torch.arange(n_rules):
            w_hat_per_rule = w_hat[i, :].unsqueeze(1).repeat(1, n_fea + 1)
            x_extra = torch.cat((torch.ones(n_smpl, 1).double(), x), 1)
            h_per_rule = torch.mul(w_hat_per_rule, x_extra).unsqueeze(0)
            h = torch.cat((h, h_per_rule), 0)
        return h
