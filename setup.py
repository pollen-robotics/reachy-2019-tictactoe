from setuptools import setup, find_packages
from os import path
import io 

here = path.abspath(path.dirname(__file__))

with io.open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='reachy_tictactoe',
    version='1.0.0',
    description='TicTacToe playground for Reachy robot',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/pollen-robotics/reachy_ticactoe',
    author='Pollen-Robotics',
    author_email='contact@pollen-robotics.com',
    packages=find_packages(exclude=['tests']),
    python_requires='>=3.5',
    install_requires=[
        'numpy',
        'zzlog',
    ],
)
