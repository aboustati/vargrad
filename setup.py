"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='vargrad',
    version='0.1a0.dev0',
    description='Variance Loss Variational Inference',
    url='https://github.com/aboustati/variance_loss_variational_inference',
    author='Ayman Boustati',
    author_email='ayman.boustati@outlook.com',
    packages=find_packages(exclude=['contrib', 'docs', 'tests', 'notebooks']),
    python_requires='>=3.7',
    install_requires=['numpy', 'scipy', 'jax', 'jaxlib', 'matplotlib', 'tqdm']
)
