# Gym Games

This is a collection of Gymnasium compatible games for reinforcement learning.

> [!NOTE]
> For Gym compatible version, please check [v1.0.4](https://github.com/qlan3/gym-games/releases/tag/v1.0.4).

For [PyGame Learning Environment](https://pygame-learning-environment.readthedocs.io/en/latest/user/games.html), the default observation is a non-visual state representation of the game. 

For [MinAtar](https://github.com/kenjyoung/MinAtar), the default observation is a visual input of the game.

## Environments

- PyGame learning environment:
  - Catcher-PLE-v0
  - FlappyBird-PLE-v0
  - Pixelcopter-PLE-v0
  - PuckWorld-PLE-v0
  - Pong-PLE-v0

- MinAtar:
  - Asterix-MinAtar-v1
  - Breakout-MinAtar-v1
  - Freeway-MinAtar-v1
  - Seaquest-MinAtar-v1
  - SpaceInvaders-MinAtar-v1

- Exploration games:
  - NChain-v1
  - LockBernoulli-v0
  - LockGaussian-v0
  - SparseMountainCar-v0
  - DiabolicalCombLock-v0

## Installation

### Gymnasium

Please read the instruction [here](https://github.com/Farama-Foundation/Gymnasium).

### Pygame

- On OSX:

      brew install sdl sdl_ttf sdl_image sdl_mixer portmidi
      pip install pygame==2.5.2

- On Ubuntu:

      sudo apt-get -y install python-pygame
      pip install pygame==2.5.2

- Others: Please read the instruction [here](http://www.pygame.org/wiki/GettingStarted#Pygame%20Installation).

### PyGame Learning Environment

    git clone https://github.com/ntasfi/PyGame-Learning-Environment.git
    cd PyGame-Learning-Environment/
    pip install -e .

## MinAtar

    pip install minatar==1.0.15

### Gym-games

    pip install git+https://github.com/qlan3/gym-games.git

## Example

Run ``python test.py``.


## Cite

Please use this bibtex to cite this repo:

```
@misc{gym-games,
  author = {Lan, Qingfeng},
  title = {Gym Compatible Games for Reinforcement Learning},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/qlan3/gym-games}}
}
```

## References

- [gym](https://github.com/openai/gym/)
- [gym-ple](https://github.com/lusob/gym-ple)
- [SRNN](https://github.com/VincentLiu3/SRNN)
- [MinAtar](https://github.com/kenjyoung/MinAtar)
- [Latent State Decoding](https://github.com/microsoft/StateDecoding)
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
