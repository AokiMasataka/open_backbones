from setuptools import setup
from setuptools import find_packages

from openback import __version__


NAME = 'openback'
VERSION = __version__


def _requires_from_file(filename):
    return open(filename).read().splitlines()


setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=_requires_from_file('./requirements.txt'),
)