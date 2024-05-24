from torch_geometric.transforms import BaseTransform
import numpy as np


class NormalizeColor(BaseTransform):
    r"""Normalizes color
    """

    def __call__(self, data):
        data.x = data.x / 255
        return data


class NormalizeFeatureToMeanStd(BaseTransform):
    r"""Standardize data.x to mean 0 and std 1
    """

    def __call__(self, data):
        mean = np.mean(np.array(data.x))
        std = np.std(np.array(data.x))
        data.x = (data.x - mean) / std

        return data

