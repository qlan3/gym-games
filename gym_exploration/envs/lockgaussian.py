import gym
import numpy as np
from gym.spaces import MultiBinary, Discrete, Box

from gym_exploration.envs.lockbernoulli import LockBernoulliEnv


class LockGaussianEnv(LockBernoulliEnv):
  ''' A (stochastic) combination lock environment
  The feature vector is hit with a random rotation and augmented with gaussian noise:
      x = s + eps where s is the one-hot encoding of the state.
  You may configure the length, dimension, and switching probability.
  Check [Provably efficient RL with Rich Observations via Latent State Decoding](https://arxiv.org/pdf/1901.09018.pdf) for a detailed description.
  '''
  def __init__(self, dimension=0, switch=0.0, noise=0.0, horizon=2):
    self.init(dimension, switch, noise, horizon)

  def init(self, dimension=0, switch=0.0, noise=0.0, horizon=2):
    super().init(horizon=horizon, dimension=dimension, switch=switch)
    self.noise = noise

  def make_obs(self, s):
    if self.noise > 0:
      new_x = np.random.normal(0, self.noise, [self.n])
    else:
      new_x = np.zeros((self.n,))
    new_x[s] += 1
    return new_x


if __name__ == '__main__':
  env = LockGaussianEnv()
  env.seed(0)
  print('Action space:', env.action_space)
  print('Obsevation space:', env.observation_space)
  print('Obsevation space high:', env.observation_space.high)
  print('Obsevation space low:', env.observation_space.low)

  for i in range(1):
    ob = env.reset()
    while True:
      action = env.action_space.sample()
      ob, reward, done, _ = env.step(action)
      print('Observation:', ob)
      print('Reward:', reward)
      print('Done:', done)
      if done:
        break
  env.close()