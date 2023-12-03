import numpy as np
from torch import nn, sigmoid
import torch


def phi_k(x, L, W):
    return sigmoid((x + (L / 2)) / W) - sigmoid((x - (L / 2)) / W)


def compute_pj(x, mu_k, K, L, W):
    # we assume that x has only one channel already
    # flatten spatial dims
    x = x.reshape(x.size(0), 1, -1)
    x = x.repeat(1, K, 1)  # construct K channels

    # apply activation functions
    return phi_k(x - mu_k, L, W)


class HistLayerBase(nn.Module):
    def __init__(self):
        super().__init__()

        self.K = 256
        self.L = 1 / self.K  # 2 / K -> if values in [-1,1] (Paper)
        self.W = self.L / 2.5

        self.mu_k = (self.L * (torch.arange(self.K) + 0.5)).view(-1, 1)


class SingleDimHistLayer(HistLayerBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        size = list(x.size())[1:]
        N = np.prod(size)
        self.mu_k = self.mu_k.to(x.device)
        pj = compute_pj(x, self.mu_k, self.K, self.L, self.W)
        return pj.sum(dim=2) / N  # We have K channels, so we sum to have K final values to represent the histogram


class JointHistLayer(HistLayerBase):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        size = list(x.size())[1:]
        N = np.prod(size)
        self.mu_k = self.mu_k.to(x.device)
        px = compute_pj(x, self.mu_k, self.K, self.L, self.W)
        py = compute_pj(y, self.mu_k, self.K, self.L, self.W)
        xy_joint_histogram = torch.matmul(px, torch.transpose(py, 1, 2)) / N
        x_histogram = px.sum(dim=2) / N
        y_histogram = py.sum(dim=2) / N

        return xy_joint_histogram, x_histogram, y_histogram


