import os
from setuptools import setup, find_packages

def read(fname):
  return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
  name='gym_games',
  version='1.0.2',
  keywords=['AI', 'Reinforcement Learning', 'Games', 'Pygame', 'MinAtar'],
  description='This is a gym version of various games for reinforcenment learning.',
  url='https://github.com/qlan3/gym-games',
  author='qlan3',
  author_email='qlan3@ualberta.ca',
  license='MIT',
  long_description=read('README.md'),
  packages=find_packages(),
  python_requires='>=3.5',
  install_requires=[
    'numpy>=1.16.4',
    'MinAtar>=1.0.4',
    'gym>=0.13.0',
    'setuptools>=41.0.1',
    'pygame>=1.9.6',
    'ple>=0.0.1'
  ],
  dependency_links=[
    'git+https://github.com/kenjyoung/MinAtar@master#egg=MinAtar-1.0.4',
    'git+https://github.com/ntasfi/PyGame-Learning-Environment@master#egg=ple-0.0.1'
  ]
)