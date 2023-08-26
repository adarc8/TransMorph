import os, glob
import torch, sys
from torch.utils.data import Dataset
from .data_utils import pkload
import matplotlib.pyplot as plt

import numpy as np


class IXIBrainDataset(Dataset):
    def __init__(self, data_path, atlas_path, transforms):
        self.paths = data_path
        self.atlas_path = atlas_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index+1]
        x, x_seg = pkload(self.atlas_path)
        y, y_seg = pkload(path)
        x, x_seg, y, y_seg = x[None, ...], x_seg[None, ...], y[None, ...], y_seg[None, ...]
        x, x_seg, y, y_seg = self.transforms([x, x_seg, y, y_seg])
        x, x_seg, y, y_seg = np.ascontiguousarray(x), np.ascontiguousarray(x_seg), np.ascontiguousarray(y), np.ascontiguousarray(y_seg)
        x, x_seg, y, y_seg = torch.from_numpy(x), torch.from_numpy(x_seg), torch.from_numpy(y), torch.from_numpy(y_seg)

        path = self.paths[index]
        y, y_seg = pkload(path)
        a = x[0, :, 75:80];
        plt.hist(a[a > 0.01].flatten(), bins=100)
        b = y[0, :, 75:80];
        plt.hist(b[b > 0.01].flatten(), bins=100);
        plt.xlim([0, 1]);
        plt.show()

        # plt.subplot(1, 2, 1)
        # plt.imshow(x[0, :, :, 80], cmap='gray')
        # plt.subplot(1, 2, 2)
        # plt.imshow(y[0, :, :, 80], cmap='gray')
        # plt.show()

        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)


class IXIBrainInferDataset(Dataset):
    def __init__(self, data_path, atlas_path, transforms):
        self.atlas_path = atlas_path
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, x_seg = pkload(self.atlas_path)
        y, y_seg = pkload(path)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)