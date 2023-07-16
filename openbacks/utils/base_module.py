import os
import logging
import torch
from torch import nn


class BaseModule(nn.Module):
    def __init__(self, init_config: dict = None, test_config: dict = None, norm_config: dict = None) -> None:
        super(BaseModule, self).__init__()
        self.init_config = init_config
        self.test_config = test_config
        self.norm_config = norm_config

        if norm_config is not None:
            assert self.norm_config['mean'].__len__() == self.norm_config['std'].__len__()
            ch = self.norm_config['mean'].__len__()
            mean = torch.tensor(self.norm_config['mean'], dtype=torch.float).view(1, ch, 1, 1)
            std = torch.tensor(self.norm_config['std'], dtype=torch.float).view(1, ch, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
            self.div = norm_config.get('div', None)
            self.norm_fn = self._norm
        else:
            self.norm_fn = self._identity
    
    def init(self, prefix: str = None, checkpoint_filter_fn=None) -> None:
        if self.init_config is not None and self.init_config.get('pretrained', False):
            pretrained = self.init_config['pretrained']
            self._init_weight(
                pretrained=pretrained,
                prefix=prefix,
                checkpoint_filter_fn=checkpoint_filter_fn
            )
    
    def _init_weight(self, pretrained: str, prefix: str, checkpoint_filter_fn=None):
        if os.path.isfile(pretrained):
            state_dict = torch.load(pretrained, map_location='cpu')
        else:
            state_dict = torch.hub.load_state_dict_from_url(pretrained, progress=False, map_location='cpu')

        if 'model' in state_dict.keys():
            state_dict = state_dict['model']

        match_keys = []
        miss_match_keys = []
        for key, value in state_dict.items():
            if checkpoint_filter_fn is not None:
                key, value = checkpoint_filter_fn(key=key, value=value, state_dict=self.state_dict)
            miss_match = True
            if key in self.state_dict().keys():
                if self.state_dict()[key].shape == value.shape:
                    self.state_dict()[key] = value
                    miss_match = False
                    match_keys.append(key)
            if miss_match:
                miss_match_keys.append(key)
        
        if self.init_config.get('log_keys', False):
            print(f'[{prefix}] match keys:')
            for match_key in match_keys:
                print('    ', match_key)
                
            print(f'[{prefix}] miss match keys:')
            for miss_match_key in miss_match_keys:
                print('    ', miss_match_key)

        logging.info(msg=f'[{prefix}] number of match keys: {match_keys.__len__()}')
        logging.info(msg=f'[{prefix}] number of miss match keys: {miss_match_keys.__len__()}')

    def _norm(self, images: torch.Tensor) -> torch.Tensor:
        if self.div is None:
            return (images - self.mean) / self.std
        else:
            return (images / self.div - self.mean) / self.std
    
    def _identity(self, images: torch.Tensor) -> torch.Tensor:
        return images
