import os
import importlib
import numpy as np
import gym
from gym import spaces
from ple import PLE

from gym_pygame.envs.base import BaseEnv


class CatcherEnv(BaseEnv):
  def __init__(self, normalize=True, display=False, **kwargs):
    self.game_name = 'Catcher'
    self.init(normalize, display, **kwargs)
    
  def get_ob_normalize(self, state):
    state_normal = self.get_ob(state)
    state_normal[0] = (state_normal[0] - 26) / 26
    state_normal[1] = (state_normal[1]) / 8
    state_normal[2] = (state_normal[2] - 26) / 26
    state_normal[3] = (state_normal[3] - 20) / 45
    return state_normal

if __name__ == '__main__':
  env = CatcherEnv(normalize=True)
  env.seed(0)
  print('Action space:', env.action_space)
  print('Action set:', env.action_set)
  print('Obsevation space:', env.observation_space)
  print('Obsevation space high:', env.observation_space.high)
  print('Obsevation space low:', env.observation_space.low)

  for i in range(1):
    ob = env.reset()
    while True:
      action = env.action_space.sample()
      ob, reward, done, _ = env.step(action)
      # env.render('rgb_array')
      env.render('human')
      print('Observation:', ob)
      print('Reward:', reward)
      print('Done:', done)
      if done:
        break
  env.close()