import torch


def cal_fc_w(x: torch.Tensor, y: torch.Tensor, para_mu):
    """
    compute W = (Y^T*Y + mu* I) * Y^T*y
    :param x:
    :param y:
    :param para_mu:
    :return:
    """
    n_fea = x.shape[1]
    w = torch.inverse(x.t().mm(x) + para_mu * torch.eye(n_fea).double()).mm(x.t().mm(y))
    return w


def mapminmax(x: torch.Tensor, l_range=-1, u_range=1):
    xmax = torch.max(x, 0)[0].unsqueeze(0)
    xmin = torch.min(x, 0)[0].unsqueeze(0)
    xmin = xmin.repeat(x.shape[0], 1)
    xmax = xmax.repeat(x.shape[0], 1)

    if (xmax == xmin).any():
        raise ValueError("some rows have no variation")
    x_proj = ((u_range - l_range) * (x - xmin) / (xmax - xmin)) + l_range

    return x_proj
