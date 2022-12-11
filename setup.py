import io
import os
from setuptools import setup, find_packages
from openbacks import __version__


NAME = 'openbacks'
DESCRIPTION = 'sementic segmentation lib'
URL = 'git@github.com:AokiMasataka/open_backbones.git'
EMAIL = None
AUTHOR = None
REQUIRES_PYTHON = '>=3.8.0'
VERSION = __version__

INSTALL_REQUIRES = ['torch']
EXTRAS_REQUIRE = {}
PACKAGES = ['openbacks']

CLASSIFIERS = [
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3 :: Only',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Visualization',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Multimedia :: Graphics',
    'Framework :: Matplotlib',
]


here = os.path.abspath(os.path.dirname(__file__))
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    include_package_data=True,
    license='MIT',
    classifiers=CLASSIFIERS
)
