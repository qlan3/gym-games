import random
import gym
import numpy as np
from gym.utils import seeding
from gym.spaces import MultiBinary, Discrete, Box


# Adapted from https://github.com/mbhenaff/PCPG/blob/main/deep_rl/component/envs.py
class DiabolicalCombLockEnv(gym.Env):
  ''' A diabolical combination lock environment: two locks
  In this task, an agent starts at an initial state s_0 (left most state), and based on its first action, transitions to one of two combination locks of length H. Each combination lock consists of a chain of length H, at the end of which are two states with high reward. At each level in the chain, 9 out of 10 actions lead the agent to a dead state (black) from which it cannot recover and lead to zero reward.
  Please check [PC-PG: Policy Cover Directed Exploration for Provable Policy Gradient Learning](http://arxiv.org/abs/2007.08459) for more details.
  '''
  def __init__(self, horizon=10, swap=0.5):
    self.init(horizon, swap)

  def init(self, horizon=10, swap=0.5):
    self.horizon = horizon
    self.n_states = 3
    self.num_actions = 10
    self.n_locks = 2
    self.optimal_reward = 5.0
    self.suboptimal_reward = 2.0
    self.locks = [OneDiabolicalCombinationLock(horizon-1), OneDiabolicalCombinationLock(horizon-1)]
    self.action_space = Discrete(self.num_actions)
    self.n_features = self.locks[0].observation_space.shape[0] + 1
    self.observation_space = Box(low=0.0, high=1.0, shape=(self.n_features,), dtype=np.float32)

  def seed(self, seed=0):
    if seed % 2 == 0:
      self.locks[0].optimal_reward = self.optimal_reward
      self.locks[1].optimal_reward = self.suboptimal_reward
    else:
      self.locks[0].optimal_reward = self.suboptimal_reward
      self.locks[1].optimal_reward = self.optimal_reward
    if hasattr(gym.spaces, 'prng'):
        gym.spaces.prng.seed(seed)
    self.locks[0].seed(seed)
    self.locks[1].seed(seed)
    return seed

  def reset(self):
    self.h = 0
    obs = np.zeros(self.observation_space.shape)
    return obs

  def step(self, action):
    assert self.n_locks == 2
    if self.h == 0:
      # In initial state, the action chooses lock
      self.lock_id = 0 if action < 5 else 1
      obs = self.locks[self.lock_id].reset()
      reward, done = 0.0, False
      info = {'state': (0, self.h, self.lock_id)}
    else:
      obs, reward, done, info = self.locks[self.lock_id].step(action)
      info['state'] = info['state'] + (self.lock_id,)
    self.h += 1
    obs = np.append(obs, self.lock_id)
    return obs, reward, done, info

  def render(self, mode='human'):
    return (self.locks[0].render(mode), self.locks[1].render(mode))

  def close(self):
    self.locks[0].close()
    self.locks[1].close()
    return None


class OneDiabolicalCombinationLock(gym.Env):
  """ One Diabolical Stochastic Combination Lock
  :param horizon: Horizon of the MDP
  :param swap: Probability for stochastic edges
  """
  def __init__(self, horizon=10, swap=0.5):
    self.init(horizon, swap)
  
  def init(self, horizon=10, swap=0.5):
    self.horizon = horizon
    self.swap = swap
    self.tolerance = 0.5
    self.optimal_reward = 5.0
    self.optimal_reward_prob = 1.0
    self.anti_shaping_reward = 0.0
    self.anti_shaping_reward2 = 1.0
    assert self.anti_shaping_reward < self.optimal_reward * self.optimal_reward_prob, \
      "Anti shaping reward shouldn't exceed optimal reward which is %r" % \
      (self.optimal_reward * self.optimal_reward_prob)
    self.num_actions = 10
    self.actions = list(range(self.num_actions))
    self.action_space = gym.spaces.Discrete(self.num_actions)
    # We encode the state type and time separately.
    # The time could be any value in 1 to horizon+1.
    # We further add noise of size horizon.
    self.obs_dim = 2 * horizon + 4
    self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)

  def seed(self, seed=0):
    self.rng, seed = seeding.np_random(seed)
    # self.rng = np.random.RandomState(seed)
    self.opt_a = self.rng.randint(low=0, high=self.action_space.n, size=self.horizon)
    self.opt_b = self.rng.randint(low=0, high=self.action_space.n, size=self.horizon)
    return seed

  def render(self, mode='human'):
    return self.make_obs(self.state)

  def make_obs(self, x):
    if x is None or self.obs_dim is None:
      return x
    else:
      v = np.zeros(self.obs_dim, dtype=np.float32)
      v[x[0]] = 1.0
      v[3 + x[1]] = 1.0
      return v

  def reset(self):
    # Start stochastically in one of the two live states
    toss_value = self.rng.binomial(1, 0.5)
    if toss_value == 0:
      self.state = [0, 0]
    elif toss_value == 1:
      self.state = [1, 0]
    else:
      raise AssertionError("Toss value can only be 1 or 0. Found %r" % toss_value)
    self.h = 0
    return self.make_obs(self.state)

  def transition(self, x, a):
    if x is None:
      raise Exception("Not in any state")
    b = self.rng.binomial(1, self.swap)
    if x[0] == 0 and a == self.opt_a[x[1]]:
      if b == 0:
        return [0, x[1] + 1]
      else:
        return [1, x[1] + 1]
    if x[0] == 1 and a == self.opt_b[x[1]]:
      if b == 0:
        return [1, x[1] + 1]
      else:
        return [0, x[1] + 1]
    else:
      return [2, x[1] + 1]

  def reward(self, x, a, next_x):
    # If the agent reaches the final live states then give it the optimal reward.
    if (x == [0, self.horizon-1] and a == self.opt_a[x[1]]) or (x == [1, self.horizon-1] and a == self.opt_b[x[1]]):
      return self.optimal_reward * self.rng.binomial(1, self.optimal_reward_prob)
    # If reaching the dead state for the first time then give it a small anti-shaping reward.
    # This anti-shaping reward is anti-correlated with the optimal reward.
    if x is not None and next_x is not None:
      if x[0] != 2 and next_x[0] == 2:
        return self.anti_shaping_reward * self.rng.binomial(1, 0.5)
      elif x[0] != 2 and next_x[0] != 2:
        return -self.anti_shaping_reward2 / (self.horizon-1)
    return 0

  def step(self, action):
    if self.state is None:
      raise Exception("Episode is not started.")
    if self.h == self.horizon:
      new_state = None
    else:
      new_state = self.transition(self.state, action)
      self.h += 1
    reward = self.reward(self.state, action, new_state)
    self.state = new_state
    # Create a dictionary containing useful debugging information
    obs = self.make_obs(self.state)
    done = self.h == self.horizon
    info = {"state": None if self.state is None else tuple(self.state)}
    return obs, float(reward), done, info

  def close(self):
    return None


def set_random_seed(seed):
  random.seed(seed)
  np.random.seed(seed)

if __name__ == '__main__':
  seed = 4
  set_random_seed(seed)
  env = DiabolicalCombLockEnv()
  env_cfg = {"horizon":5, "swap":0.5}
  env.init(**env_cfg)
  env.seed(seed)
  env.action_space.np_random.seed(seed)
  env.locks[0].action_space.np_random.seed(seed)
  env.locks[1].action_space.np_random.seed(seed)
  
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
      print('Obser:', ob)
      print('action:', action)
      print('Reward:', reward)
      print('Done:', done)
      if done:
        break
  env.close()