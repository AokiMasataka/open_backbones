from torch import nn
from copy import deepcopy


def build_activation(config):
    copy_config = deepcopy(config)
    _type = copy_config.pop('type')
    activation = getattr(nn, _type)
    return activation(**copy_config)


def build_norm(config):
    copy_config = deepcopy(config)
    _type = copy_config.pop('type')
    if _type == 'LN':
        _type = 'LayerNorm'
    
    activation = getattr(nn, _type)
    return activation(**copy_config)
