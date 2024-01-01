import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import logging
import torch
from torch import nn


class BaseModule(nn.Module):
	def __init__(self, init_config: dict) -> None:
		super().__init__()
		self.init_config = init_config
	
	def _init(self, prefix: str) -> None:
		if self.init_config is not None:
			self._load_pretrained(prefix=prefix)

	def _load_pretrained(self, prefix: str) -> None:
		pretrained = self.init_config['pretrained']

		if os.path.isfile(pretrained):
			state_dict = torch.load(pretrained, map_location='cpu')
		else:
			state_dict = torch.hub.load_state_dict_from_url(pretrained, progress=False, map_location='cpu')

		if 'model' in state_dict.keys():
			state_dict = state_dict['model']
		
		match_keys = []
		miss_match_keys = []
		for key, value in state_dict.items():
			miss_match = True
			if key in self.state_dict().keys():
				if self.state_dict()[key].shape == value.shape:
					self.state_dict()[key] = value
					miss_match = False
					match_keys.append(key)
			if miss_match:
				miss_match_keys.append(key)

		if self.init_config.get('log_keys', False):
			logging.info(msg=f'[{prefix}] match keys:')
			for match_key in match_keys:
				logging.info(msg='-' + match_key)
				
			logging.info(msg=f'[{prefix}] miss match keys:')
			for miss_match_key in miss_match_keys:
				logging.info(msg='-'+miss_match_key)

		logging.info(msg=f'[{prefix}] number of match keys: {match_keys.__len__()}')
		logging.info(msg=f'[{prefix}] number of miss match keys: {miss_match_keys.__len__()}')
