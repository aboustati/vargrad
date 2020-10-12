"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='vargrad_experiments',
    version='0.1a0.dev0',
    description='Experiments for Variance Loss Variational Inference',
    url='https://github.com/aboustati/variance_loss_variational_inference',
    author='Ayman Boustati',
    author_email='ayman.boustati@outlook.com',
    python_requires='>=3.7'
)
