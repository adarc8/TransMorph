import os, glob
from typing import Optional

import torch, sys
from torch.utils.data import Dataset
from .data_utils import pkload
import matplotlib.pyplot as plt

import numpy as np


class JHUBrainDataset(Dataset):
    def __init__(self, data_path, transforms, atlas_path: Optional = None):
        self.paths = data_path
        self.transforms = transforms
        self._atlas_path = atlas_path

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        if self._atlas_path is not None:
            x, x_seg = pkload(self._atlas_path)
            y, y_seg = pkload(path)
        else:
            x, x_seg = pkload(path)
            y, y_seg = pkload(self._get_random_path())

        x, x_seg, y, y_seg = self._add_batch_dimension(x, x_seg, y, y_seg)

        x, y, x_seg, y_seg = self.transforms([x, y, x_seg, y_seg])
        # [Bsize,channelsHeight,,Width,Depth]
        x, x_seg, y, y_seg = self._np_to_torch(x, x_seg, y, y_seg)
        return x, y, x_seg, y_seg

    def _get_random_path(self):
        path = self.paths[np.random.randint(0, len(self.paths))]
        return path

    def _np_to_torch(self, x, x_seg, y, y_seg):
        x, y, x_seg, y_seg = np.ascontiguousarray(x), np.ascontiguousarray(y), \
            np.ascontiguousarray(x_seg), np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, x_seg, y, y_seg

    def _add_batch_dimension(self, x, x_seg, y, y_seg):
        x, y, x_seg, y_seg = x[None, ...], y[None, ...], x_seg[None, ...], y_seg[None, ...]
        return x, x_seg, y, y_seg


