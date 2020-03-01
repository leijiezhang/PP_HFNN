import torch.nn as nn
import torch
import abc


class LossFunc(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    @abc.abstractmethod
    def forward(self, yhat, y: torch.Tensor):
        loss = []
        return loss


class RMSELoss(LossFunc):
    def __init__(self, eps=1e-6):
        super().__init__()
        # self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y, yhat):
        loss = torch.sqrt(torch.norm(yhat - y).pow(2) / (y.shape[0]))
        return loss


class NRMSELoss(LossFunc):
    def __init__(self, eps=1e-6):
        super().__init__()
        # self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y, yhat):
        loss = torch.sqrt(torch.norm(yhat - y).pow(2) / (y.shape[0])) / y.mean()
        return loss


class MSELoss(LossFunc):
    def __init__(self, eps=1e-6):
        super().__init__()
        # self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y, yhat):
        loss = torch.norm(yhat - y).pow(2) / (y.shape[0])
        return loss


class Map(LossFunc):
    def __init__(self, eps=1e-6):
        super().__init__()
        # self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y, yhat):
        y_ones = torch.ones(y.shape)
        y_zeros = torch.zeros(y.shape)
        acc_num = torch.where(yhat == y, y_ones, y_zeros).sum()
        acc = acc_num / y.shape[0]
        return acc


class LikelyLoss(LossFunc):
    """
    todo: used for calculate the loss of classification task with one output node
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        # self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y, yhat):
        yhat = torch.round(yhat)
        yhat = torch.where(yhat > max(y), max(y), yhat)
        y_ones = torch.ones(y.shape)
        y_zeros = torch.zeros(y.shape)
        acc_num = torch.where(yhat == y, y_ones, y_zeros).sum()
        acc = (acc_num / y.shape[0])
        return acc

