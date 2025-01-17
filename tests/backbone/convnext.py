import torch

from openback import build_backbone
from openback.utils import load_config_file


def test_convnext():
	config_path = './configs/convnext/convnext_tiny.py'
	config, _ = load_config_file(path=config_path)
	config['backbone']['init_config']['log_keys'] = True

	model = build_backbone(config=config['backbone'])
	x = torch.randint(low=0, high=256, size=(2, 3, 256, 256), dtype=torch.float)

	with torch.no_grad():
		y = model(x)
	
	print(y[0].shape[3] == 64 and y[3].shape[3] == 8)


if __name__ == '__main__':
	test_convnext()
