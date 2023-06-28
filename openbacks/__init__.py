from . import layers
from . import backbones
from .utils import build_activation, build_norm
from .builder import build_backbone


VERSION = (1, 0, 0, 'post0')
__version__ = '.'.join(map(str, VERSION))
