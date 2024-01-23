import numpy as np

import gymnasium as gym
from gymnasium import spaces


# Adapted from https://raw.githubusercontent.com/microsoft/StateDecoding/master/LockBernoulli.py
class LockBernoulliEnv(gym.Env):
  ''' A (stochastic) combination lock environment
  You may configure the length, dimension, and switching probability.
  Check [Provably efficient RL with Rich Observations via Latent State Decoding](https://arxiv.org/pdf/1901.09018.pdf) for a detailed description.
  '''
  def __init__(self, dimension=0, switch=0.0, horizon=2):
    self.init(dimension, switch, horizon)

  def init(self, dimension=0, switch=0.0, horizon=2):
    self.dimension = dimension
    self.switch = switch
    self.horizon = horizon
    self.n = self.dimension+3
    self.observation_space = spaces.Box(low=0, high=1, shape=(self.n,), dtype=np.uint8)
    self.action_space = spaces.Discrete(4)

  def _get_obs(self, s):
    new_x = np.zeros((self.n,), dtype=np.uint8)
    new_x[s] = 1
    new_x[3:] = self.np_random.binomial(1, 0.5, (self.dimension,))
    return new_x

  def reset(self, seed=None, options=None):
    # We need the following line to seed self.np_random
    super().reset(seed=seed)
    if seed is not None:
      self.opt_a = self.np_random.integers(low=0, high=self.action_space.n, size=self.horizon, dtype=np.uint8)
      self.opt_b = self.np_random.integers(low=0, high=self.action_space.n, size=self.horizon, dtype=np.uint8)  
    self.h = 0
    self.state = 0
    obs = self._get_obs(self.state)
    return obs, {}

  def step(self, action):
    assert self.h < self.horizon, 'Exceeded horizon!'
    if self.h == self.horizon-1:
      terminated = True
      r = self.np_random.binomial(1, 0.5)
      if self.state == 0 and action == self.opt_a[self.h]:
        next_state = 0
      elif self.state == 0 and action == (self.opt_a[self.h]+1) % 4:
        next_state = 1
      elif self.state == 1 and action == self.opt_b[self.h]:
        next_state = 1
      elif self.state == 1 and action == (self.opt_b[self.h]+1) % 4:
        next_state = 0
      else:
        next_state, r = 2, 0
    else:
      r = 0
      terminated = False
      ber = self.np_random.binomial(1, self.switch)
      if self.state == 0: # state A
        if action == self.opt_a[self.h]:
          next_state = ber
        elif action == (self.opt_a[self.h]+1) % 4:
          next_state = 1 - ber
        else:
          next_state = 2
      elif self.state == 1: # state B
        if action == self.opt_b[self.h]:
          next_state = 1 - ber
        elif action == (self.opt_b[self.h]+1) % 4:
          next_state = ber
        else:
          next_state = 2
      else: # state C
        next_state = 2

    self.h += 1
    self.state = next_state
    obs = self._get_obs(self.state)
    return obs, r, terminated, False, {}

  def render(self, mode="human"):
    print(f'{chr(self.state+65)}{self.h}')

  def close(self):
    return None


if __name__ == '__main__':
  seed = 4
  env = LockBernoulliEnv()
  env_cfg = {"horizon":10, "dimension":10, "switch":0.1}
  env.init(**env_cfg)
  obs, info = env.reset(seed)
  env.action_space.seed(seed)
  env.observation_space.seed(seed)

  print('Action space:', env.action_space)
  print('Obsevation space:', env.observation_space)
  print('Obsevation space high:', env.observation_space.high)
  print('Obsevation space low:', env.observation_space.low)

  for i in range(2):
    obs, info = env.reset(i)
    env.action_space.seed(i)
    env.observation_space.seed(i)
    while True:
      action = env.action_space.sample()
      obs, reward, terminated, _, _ = env.step(action)
      print('Observation:', obs)
      print('action:', action)
      print('Reward:', reward)
      print('Done:', terminated)
      if terminated:
        break
  env.close()