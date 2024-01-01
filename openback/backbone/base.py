import torch
from ..utils import BaseModule


class Normlizer(torch.nn.Module):
	def __init__(self, mean: list, std: list, div: float) -> None:
		super().__init__()

		if mean is not None or std is not None:
			assert mean.__len__() == std.__len__()
			ch = mean.__len__()
			_mean = torch.tensor(mean, dtype=torch.float).view(1, ch, 1, 1)
			_std = torch.tensor(std, dtype=torch.float).view(1, ch, 1, 1)
			self.register_buffer('_mean', _mean)
			self.register_buffer('_std', _std)
		else:
			self._mean = None
			self._std = None
		self._div = div
	
	@torch.no_grad
	def __call__(self, x: torch.Tensor) -> torch.Tensor:
		if self._div is not None:
			x = x / self._div
		
		if self._mean is None:
			return x
		else:
			return (x - self._mean) / self._std


class BaseBackBone(BaseModule):
	def __init__(self, init_config: dict, norm_config: dict) -> None:
		super().__init__(init_config=init_config)
		self.norm_config = norm_config

		if norm_config is not None:
			self.norm = Normlizer(
				mean=norm_config['mean'],
				std=norm_config['std'],
				div=norm_config.get('div', None)
			)
		else:
			self.norm = lambda x: x
