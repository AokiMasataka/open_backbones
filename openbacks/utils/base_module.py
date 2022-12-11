import logging
import torch
from torch import nn


class BaseModule(nn.Module):
    def __init__(self, init_confg=None):
        super(BaseModule, self).__init__()
        if init_confg is not None:
            assert isinstance(init_confg, dict)
        else:
            init_confg = dict()
        self.init_config = init_confg

    def init(self, checkpoint_filter_fn=None):
        if self.init_config.get('pretrained', False):
            state_dict = torch.hub.load_state_dict_from_url(self.init_config['pretrained'], progress=False, map_location='cpu')
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
            
            logging.info(msg='match keys:')
            if self.init_config.get('print_match_key', False):
                print('match keys:')
            for match_key in match_keys:
                logging.info(msg=f'    {match_key}')
                if self.init_config.get('print_match_key', False):
                    print('    ', match_key)
                
            logging.info(msg='miss match keys:')
            if self.init_config.get('print_miss_match_key', False):
                print('miss match keys:')
            for miss_match_key in miss_match_keys:
                logging.info(msg=f'    {miss_match_key}')
                if self.init_config.get('print_miss_match_key', False):
                    print('    ', miss_match_key)

            logging.info(msg=f'number of match keys: {match_keys.__len__()}')
            logging.info(msg=f'number of miss match keys: {miss_match_keys.__len__()}')
            print(f'number of match keys: {match_keys.__len__()}')
            print(f'number of miss match keys: {miss_match_keys.__len__()}')