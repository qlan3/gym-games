import gym
import numpy as np
from gym.utils import seeding
from gym.spaces import MultiBinary, Discrete, Box


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
    self.observation_space = Box(low=0.0, high=1.0, shape=(self.n,), dtype=np.float32)
    self.action_space = Discrete(4)
    self.seed()

  def reset(self):
    self.h = 0
    self.state = 0
    obs = self.make_obs(self.state)
    return (obs)

  def make_obs(self, s):
    new_x = np.zeros((self.n,))
    new_x[s] = 1
    new_x[3:] = self.np_random.binomial(1, 0.5, (self.dimension,))
    return new_x

  def step(self,action):
    assert self.h < self.horizon, 'Exceeded horizon!'
    if self.h == self.horizon-1:
      done = True
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
      done = False
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
    obs = self.make_obs(self.state)
    return obs, r, done, {}

  def render(self, mode='human'):
    print(f'{chr(self.state+65)}{self.h}')

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    self.opt_a = self.np_random.randint(low=0, high=self.action_space.n, size=self.horizon)
    self.opt_b = self.np_random.randint(low=0, high=self.action_space.n, size=self.horizon)
    if hasattr(gym.spaces, 'prng'):
      gym.spaces.prng.seed(seed)
    return seed

  def close(self):
    return 0


if __name__ == '__main__':
  env = LockBernoulliEnv()
  env.seed(0)
  env_cfg = {"horizon":10, "dimension":10, "switch":0.1}
  env.init(**env_cfg)
  print('Action space:', env.action_space)
  print('Obsevation space:', env.observation_space)
  try:
    print('Obsevation space high:', env.observation_space.high)
    print('Obsevation space low:', env.observation_space.low)
  except:
    pass

  for i in range(1):
    ob = env.reset()
    print('Observation:', ob)
    while True:
      action = env.action_space.sample()
      ob, reward, done, _ = env.step(action)
      print('Observation:', ob)
      print('Reward:', reward)
      print('Done:', done)
      if done:
        break
  env.close()