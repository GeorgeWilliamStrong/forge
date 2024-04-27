from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='forge',
    version='1.0',
    description='A full-waveform inversion framework built in PyTorch',
    long_description='A full-waveform inversion framework built in PyTorch',
    author='George Strong',
    author_email='geowstrong@gmail.com',
    license='AGPL-3.0',
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=required)
