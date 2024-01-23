import numpy as np

import gymnasium as gym
from gymnasium import spaces


# Adapted from https://github.com/facebookresearch/RandomizedValueFunctions/blob/master/qlearn/envs/nchain.py
class NChainEnv(gym.Env):
  ''' N-Chain environment
  The environment consists of a chain of N states and the agent always starts in state s2,
  from where it can either move left or right.
  In state s1, the agent receives a small reward of r = 0.001 and a larger reward r = 1 in state sN.
  Check [Deep Exploration via Bootstrapped DQN](https://papers.nips.cc/paper/6501-deep-exploration-via-bootstrapped-dqn.pdf) for a detailed description.
  '''
  def __init__(self, n=10):
    self.state = 1  # Start at state s2
    self.action_space = spaces.Discrete(2)
    self.init(n)
    
  def init(self, n=10):
    self.n = n
    self.observation_space = spaces.Box(low=0, high=1, shape=(self.n,), dtype=np.uint8)
    self.max_steps = n+8
  
  def _get_obs(self, v):
    return (v <= self.state).astype(np.uint8)

  def reset(self, seed=None, options=None):
    # We need the following line to seed self.np_random
    super().reset(seed=seed)
    v = np.arange(self.n)
    self.state = 1
    self.steps = 0
    return self._get_obs(v), {}
  
  def reward(self, s, a):
    if s == self.n-1 and a == 1:
      return 1.0  
    elif s == 0 and a == 0:
      return 0.001
    else:
      return 0

  def step(self, action):
    assert self.action_space.contains(action)
    v = np.arange(self.n)
    
    r = self.reward(self.state, action)
    if action == 1:
      if self.state != self.n - 1:
        self.state += 1
    else:
      if self.state != 0:
        self.state -= 1
    self.steps += 1
    if self.steps >= self.max_steps:
      terminated = True
    else:
      terminated = False
    return self._get_obs(v), r, terminated, False, {}

  def render(self, mode="human"):
    pass

  def close(self):
    return None
  

if __name__ == '__main__':
  seed = 4
  env = NChainEnv()
  print('Action space:', env.action_space)
  print('Obsevation space:', env.observation_space)
  print('Obsevation space high:', env.observation_space.high)
  print('Obsevation space low:', env.observation_space.low)

  cfg = {'n':5}
  env.init(**cfg)
  print('New obsevation space:', env.observation_space)
  print('New Obsevation space high:', env.observation_space.high)
  print('New Obsevation space low:', env.observation_space.low)
  
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