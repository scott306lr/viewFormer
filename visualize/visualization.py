import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from ..utils.methods import absmax, absmax_reduce


def outlier_heatmap(data, kernel_size=1, axis=None, ax=None, *args, **kwargs):
    if ax is None:
        # keep the aspect ratio of the original data
        fig, self_ax = plt.subplots(
            1, 1, figsize=(data.shape[1], data.shape[0]))
    ax = ax or self_ax

    matrix = absmax_reduce(data, kernel_size=kernel_size, axis=axis)
    g = sns.heatmap(matrix, ax=ax, **kwargs)

    y_ticks = np.arange(0, matrix.shape[0], math.ceil(matrix.shape[0]/100)*10)
    x_ticks = np.arange(0, matrix.shape[1], math.ceil(matrix.shape[1]/100)*10)

    ax.set_yticks(y_ticks, labels=y_ticks*kernel_size)
    ax.set_xticks(x_ticks, labels=x_ticks*kernel_size)
    return g
