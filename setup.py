from setuptools import setup, find_packages
from openbacks import __version__


NAME = 'openbacks'
VERSION = __version__,
REQUIRES_PYTHON = '>=3.10.0'


def requirements_from_file(file_name):
    return open(file_name).read().splitlines()


print(VERSION)
setup(
    name=NAME,
    version="0.0.1",
    python_requires=REQUIRES_PYTHON,
    packages=find_packages(),
    install_requires=requirements_from_file('requirements.txt'),
)