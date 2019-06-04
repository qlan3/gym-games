# Gym PyGame
This is a gym version of PyGame Learning Environment (PLE). The default observation is a non-visual state representation of the game.

## Environments

- Catcher-v0
- FlappyBird-v0
- Pixelcopter-v0
- PuckWorld-v0
- PongPLE-v0 (PLE version of Pong)

A detailed explaination of games can be find [here](https://pygame-learning-environment.readthedocs.io/en/latest/user/games.html).

## Installation

### Gym

Please read the instruction [here](https://github.com/openai/gym).

### PyGame

- On OSX:

    brew install sdl sdl_ttf sdl_image sdl_mixer portmidi
    pip3 install pygame

- On Ubuntu:

    sudo apt-get install -y python3-pygame

- Others: Please read the instruction [here](http://www.pygame.org/wiki/GettingStarted#Pygame%20Installation).


### PLE

    git clone https://github.com/ntasfi/PyGame-Learning-Environment.git
    cd PyGame-Learning-Environment/
    pip install -e .

### gym-pygame

    git clone https://github.com/qlan3/gym-pygame
    cd gym-pygame/
    pip install -e .

## Example
Run ``python test.py``.

## References
- [gym](https://github.com/openai/gym/tree/master/)
- [gym-ple](https://github.com/lusob/gym-ple)
- [SRNN](https://github.com/VincentLiu3/SRNN)