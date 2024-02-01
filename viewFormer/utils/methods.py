import numpy as np
from skimage.measure import block_reduce


def absmax(data, axis=None):
    p_data = data.max(axis=axis)
    n_data = data.min(axis=axis)
    return np.where(abs(p_data) > abs(n_data), p_data, n_data)


def absmax_reduce(data, kernel_size=1, axis=None):
    if axis is not None:
        matrix = absmax(data, axis=axis)
    else:
        matrix = data
    p_data = block_reduce(matrix, block_size=kernel_size, func=np.max)
    n_data = block_reduce(matrix, block_size=kernel_size, func=np.min)
    return np.where(abs(p_data) > abs(n_data), p_data, n_data)


def match_string(string, match_list, match_prefix=False, match_suffix=False):
    if match_prefix:
        return any(string.startswith(s) for s in match_list)
    elif match_suffix:
        return any(string.endswith(s) for s in match_list)
    else:
        return any(s in string for s in match_list)


def get_model_layers(model, match_names=None, match_types=None, prefix=''):
    matching_layers = []
    for name, module in model.named_modules():
        if match_names is None or match_string(name, match_names):
            if match_types is None or match_string(type(module).__name__, match_types):
                matching_layers.append((f'{prefix}{name}', module))
    return matching_layers


def get_layer_weights(layer, match_names=None, match_types=None, prefix=''):
    matching_weights = []
    for name, param in layer.named_parameters():
        if match_names is None or match_string(name, match_names):
            if match_types is None or match_string(type(param).__name__, match_types):
                matching_weights.append((f'{prefix}{name}', param))
    return matching_weights
