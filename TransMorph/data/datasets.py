import os, glob
from typing import Optional

import torch, sys
from torch.utils.data import Dataset
from .data_utils import pkload
import matplotlib.pyplot as plt

import numpy as np


class JHUBrainDataset(Dataset):
    def __init__(self, data_path, transforms, atlas_path: Optional = None,
                 n_classes: int = 4, short_dataset: bool = False):
        self.paths = data_path
        self.transforms = transforms
        self._atlas_path = atlas_path
        self._n_classes = n_classes
        self._short_dataset = short_dataset


    def __len__(self):
        return 10 if self._short_dataset else len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        if self._atlas_path is not None:
            # Brain2Atlas
            x, x_seg = pkload(self._atlas_path)
            y, y_seg = pkload(path)
        else:
            # Brain2Brain
            x, x_seg = pkload(path)  # np array should be shape (160, 192, 224)
            y, y_seg = pkload(self._get_random_path())

        x = x[:, 80:120]
        y = y[:, 80:120]

        # x_seg = self._get_GMM_seg(x, n_classes=self._n_classes)
        # y_seg = self._get_GMM_seg(y, n_classes=self._n_classes)

        # new_seg = np.zeros_like(x_seg)
        # new_seg[(x_seg == 2) + (x_seg == 41)] = 1
        # new_seg[(x_seg == 3) + (x_seg == 42)] = 2
        # new_seg[x_seg == 0] = 3
        # new_seg[new_seg == 0] = 4

        # for i in range(255):
        #     class_i = i;
        #     m = (x_seg[:,100] == class_i).mean()
        #     if m> 0.03:
        #         print("class {} has {}%".format(class_i, m*100))

        x = (x - x.min()) / (x.max() - x.min())
        y = (y - y.min()) / (y.max() - y.min())

        x_seg = np.digitize(x, np.linspace(0, 1, self._n_classes + 1)) - 1
        y_seg = np.digitize(y, np.linspace(0, 1, self._n_classes + 1)) - 1

        x, x_seg, y, y_seg = self._add_batch_dimension(x, x_seg, y, y_seg)
        # x, x_seg, y, y_seg = self.transforms([x, x_seg, y, y_seg])
        # x, x_seg, y, y_seg = self._np_to_torch(x, x_seg, y, y_seg)

        return x, x_seg, y, y_seg

    def _get_random_path(self):
        path = self.paths[np.random.randint(0, len(self.paths))]
        return path

    def _np_to_torch(self, x, x_seg, y, y_seg):
        # [Bsize,channelsHeight,,Width,Depth]
        x, x_seg, y, y_seg = np.ascontiguousarray(x), np.ascontiguousarray(x_seg), \
            np.ascontiguousarray(y), np.ascontiguousarray(y_seg)
        x, x_seg, y, y_seg = torch.from_numpy(x), torch.from_numpy(x_seg), torch.from_numpy(y), torch.from_numpy(y_seg)
        return x, x_seg, y, y_seg

    def _add_batch_dimension(self, x, x_seg, y, y_seg):
        x, y, x_seg, y_seg = x[None, ...], y[None, ...], x_seg[None, ...], y_seg[None, ...]
        return x, x_seg, y, y_seg

    def _get_GMM_seg(self, x: np.ndarray, n_classes: int):
        from sklearn.mixture import GaussianMixture
        original_shape = x.shape
        x = x.reshape(-1, 1)
        gmm = GaussianMixture(n_components=n_classes, covariance_type='full', random_state=0).fit(x)
        labels = gmm.predict(x)
        return labels.reshape(original_shape)


