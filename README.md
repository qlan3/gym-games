# Gym Games
This is a gym version of various games for reinforcenment learning. The default observation is a non-visual state representation of the game.

## Environments

- [PyGame learning environment](https://pygame-learning-environment.readthedocs.io/en/latest/user/games.html):
  - Catcher-PLE-v0
  - FlappyBird-PLE-v0
  - Pixelcopter-PLE-v0
  - PuckWorld-PLE-v0
  - Pong-PLE-v0

- [MinAtar](https://github.com/kenjyoung/MinAtar):
  - Asterix-MinAtar-v0
  - Breakout-MinAtar-v0
  - Freeway-MinAtar-v0
  - Seaquest-MinAtar-v0
  - Space_invaders-MinAtar-v0

## Installation

### Gym

Please read the instruction [here](https://github.com/openai/gym).

### PyGame

- On OSX:

    brew install sdl sdl_ttf sdl_image sdl_mixer portmidi
    pip install pygame

- On Ubuntu:

    sudo apt-get -y install python-pygame
    pip install pygame

- Others: Please read the instruction [here](http://www.pygame.org/wiki/GettingStarted#Pygame%20Installation).

### PLE

    git clone https://github.com/ntasfi/PyGame-Learning-Environment.git
    cd PyGame-Learning-Environment/
    pip install -e .

## MinAtar

    pip install git+https://github.com/kenjyoung/MinAtar.git

### gym-games

    pip install git+https://github.com/qlan3/gym-games.git

## Example
Run ``python test.py``.

## References
- [gym](https://github.com/openai/gym/tree/master/)
- [gym-ple](https://github.com/lusob/gym-ple)
- [SRNN](https://github.com/VincentLiu3/SRNN)
- [MinAtar](https://github.com/kenjyoung/MinAtar)