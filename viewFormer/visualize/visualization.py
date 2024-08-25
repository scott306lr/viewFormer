import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from ..utils.methods import absmax, absmax_reduce


def outlier_heatmap(data, kernel_size=1, axis=None, ax=None, *args, **kwargs):
    if ax is None:
        # keep the aspect ratio of the original data
        fig, self_ax = plt.subplots(
            1, 1, figsize=(data.shape[1], data.shape[0]))
    ax = ax or self_ax

    matrix = absmax_reduce(data, kernel_size=kernel_size, axis=axis)
    # g = sns.heatmap(matrix, ax=ax, **kwargs)
    # matplot lib heatmap
    g = ax.imshow(matrix, **kwargs)

    y_ticks = np.arange(0, matrix.shape[0], math.ceil(matrix.shape[0]/100)*10)
    x_ticks = np.arange(0, matrix.shape[1], math.ceil(matrix.shape[1]/100)*10)

    ax.set_yticks(y_ticks, labels=y_ticks*kernel_size)
    ax.set_xticks(x_ticks, labels=x_ticks*kernel_size)

    # draw color bar, pad is the distance between color bar and plot
    m = cm.ScalarMappable(**kwargs)
    m.set_array(matrix)
    ax.figure.colorbar(m, ax=ax, shrink=0.6, pad=0.1).ax.tick_params(labelsize=14)

    # set axis font size
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # set box aspect
    ax.set_aspect('auto')

    return g


def abs_outlier_tensor(w, ax=None, *args, **kwargs):
    if ax is None:
        fig, self_ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(10, 10))
    ax = ax or self_ax

    # draw 3d surface plot with coolwarm color map
    x = np.arange(w.shape[1])  # Dims
    y = np.arange(w.shape[0])  # Seqlen
    xpos, ypos = np.meshgrid(x, y, copy=False)
    zpos = np.abs(w)
    surf = ax.plot_surface(xpos, ypos, zpos, antialiased=False, **kwargs)

    # set labels
    ax.set_xlabel('Out Dims', fontsize=18, labelpad=10)
    ax.set_ylabel('SeqLen', fontsize=18, labelpad=10)
    ax.set_zlabel('Absolute Value', fontsize=18, labelpad=10)

    # set axis font size
    ax.tick_params(axis='both', which='major', labelsize=14)

    # draw color bar, pad is the distance between color bar and plot
    m = cm.ScalarMappable(**kwargs)
    m.set_array(w)
    ax.figure.colorbar(m, ax=ax, shrink=0.6, pad=0.1).ax.tick_params(labelsize=14)
    

    # set box aspect
    # w_h = w.shape[1] / w.shape[0]
    ax.set_box_aspect([1, 1, 1])

    return surf