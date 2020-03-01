import abc
import torch


class FnnSolveBase(object):
    def __init__(self):
        self.para_mu = None
        self.h: torch.Tensor = None
        self.y: torch.Tensor = None

    @abc.abstractmethod
    def solve(self):
        w_optimal = []
        return w_optimal


class FnnSolveReg(FnnSolveBase):
    def __init__(self):
        super(FnnSolveReg, self).__init__()

    def solve(self):
        """
        todo: fnn solver for regression problem
        """
        n_rule = self.h.shape[0]
        n_smpl = self.h.shape[1]
        n_fea = self.h.shape[2]
        h_cal = self.h.permute((1, 0, 2))  # N * n_rules * (d + 1)
        h_cal = h_cal.reshape(n_smpl, n_rule * n_fea)  # squess the last dimension
        w_comb_optimal = torch.inverse(h_cal.t().mm(h_cal) +
                                       self.para_mu * torch.eye(n_rule * n_fea).double()).mm(h_cal.t().mm(self.y))
        w_comb_optimal = w_comb_optimal.permute((1, 0))
        w_optimal = w_comb_optimal.reshape(self.y.shape[1], n_rule, n_fea)

        return w_optimal


class FnnSolveAO(FnnSolveBase):
    def __init__(self):
        super(FnnSolveAO, self).__init__()

    def solve(self):
        """
        todo: fnn solver for regression problem
        """
        n_rule = self.h.shape[0]
        n_smpl = self.h.shape[1]
        n_fea = self.h.shape[2]
        h_cal = self.h.permute((1, 0, 2))  # N * n_rules * (d + 1)
        h_cal = h_cal.reshape(n_smpl, n_rule * n_fea)  # squess the last dimension

        loss = 100
        th_run = 0.0001
        while loss > th_run:
            w_comb_optimal = torch.inverse(h_cal.t().mm(h_cal) +
                                           self.para_mu * torch.eye(n_rule * n_fea).double()).mm(h_cal.t().mm(self.y))
            w_comb_optimal = w_comb_optimal.permute((1, 0))
            w_optimal = w_comb_optimal.reshape(self.y.shape[1], n_rule, n_fea)

        return w_optimal


class FnnSolveCls(FnnSolveBase):
    def __init__(self):
        super(FnnSolveCls, self).__init__()

    def solve(self):
        """
        todo: fnn solver for classification problem
        """
        n_rule = self.h.shape[0]
        n_smpl = self.h.shape[1]
        n_fea = self.h.shape[2]
        n_output = self.y.shape[1]
        h_cal = self.h.permute((1, 0, 2))  # N * n_rules * (d + 1)
        len_w = n_rule * n_fea
        h_cal = h_cal.reshape(n_smpl, len_w)  # squess the last dimension

        w_optimal = torch.zeros(len_w, n_output).double()

        sh_cal = 0.001  # initiate the threshold
        w_loss = 1  # initiate the loss of W
        w_loss_list = []

        for i in torch.arange(n_output):
            w_tmp = w_optimal[:, i].unsqueeze(1).double()
            y_tmp = self.y[:, i].unsqueeze(1).double()
            s = torch.ones(n_smpl, 1).double()
            z = torch.ones(n_smpl, 1).double()
            while w_loss > sh_cal:
                w_old = w_tmp.clone()
                s_diag = torch.diag(s.squeeze())
                w_tmp = torch.inverse(h_cal.t().mm(s_diag).mm(h_cal) +
                                      self.para_mu * torch.eye(len_w)).mm(h_cal.t().mm(s_diag).mm(z))
                mu_cal = torch.sigmoid(h_cal.mm(w_tmp))
                s = torch.mul(mu_cal, (torch.ones(n_smpl, 1) - mu_cal))
                z = (h_cal.mm(w_tmp) + torch.div((y_tmp - mu_cal), s))

                w_loss = torch.norm((w_tmp - w_old))
                w_loss_list.append(w_loss)
            w_optimal[:, i] = w_tmp.squeeze()
        w_optimal = w_optimal.permute((1, 0))
        w_optimal = w_optimal.reshape(n_output, n_rule, n_fea)

        return w_optimal
