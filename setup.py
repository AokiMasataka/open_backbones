from setuptools import setup
from setuptools import find_packages


NAME = 'openback'
VERSION = '0.1.0'


def _requires_from_file(filename):
    return open(filename, encoding='UTF-8').read().splitlines()


setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=_requires_from_file('./requirements.txt'),
)
