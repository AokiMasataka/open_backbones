import torch

from openback.backbone import ResNet
from openback.utils import load_config_file


def test_resnet():
	config_path = './configs/resnet/resnet34.py'
	config, _ = load_config_file(path=config_path)

	_ = config['backbone'].pop('type')
	model = ResNet(**config['backbone'])
	x = torch.rand(2, 3, 256, 256)

	with torch.no_grad():
		y = model(x)
	
	print(y[0].shape[3] == 64 and y[4].shape[3] == 4)


if __name__ == '__main__':
	test_resnet()
