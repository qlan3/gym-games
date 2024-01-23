from setuptools import setup, find_packages

with open('README.md', 'r') as f:
  long_description = f.read()

setup(
  name='gym-games',
  version='2.0.0',
  keywords=['AI', 'Reinforcement Learning', 'Games', 'Pygame', 'MinAtar'],
  description='A gymnasium version of various games for reinforcement learning.',
  url='https://github.com/qlan3/gym-games',
  author='qlan3',
  author_email='qlan3@ualberta.ca',
  license='MIT',
  long_description=long_description,
  long_description_content_type='text/markdown',
  packages=find_packages(),
  python_requires='>=3.7',
  install_requires=[
    'MinAtar==1.0.15',
    'gymnasium==0.28.1',
    'setuptools>=69.0.3',
    'pygame==2.5.2',
    'ple==0.0.1'
  ],
  dependency_links=[
    'git+https://github.com/kenjyoung/MinAtar@master#egg=MinAtar-1.0.15',
    'git+https://github.com/ntasfi/PyGame-Learning-Environment@master#egg=ple-0.0.1'
  ]
)
