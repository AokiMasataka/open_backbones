from .utils import Registry


BACKBONES = Registry(name='backbones')


def build_backbone(config):
    return BACKBONES.build(config=config)