import numpy as np

import gymnasium as gym
from gymnasium import spaces


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
    self.action_space = spaces.Discrete(self.num_actions)
    self.n_features = self.locks[0].observation_space.shape[0] + 1
    self.observation_space = spaces.Box(low=0, high=1, shape=(self.n_features,), dtype=np.uint8)

  def reset(self, seed=None, options=None):
    # We need the following line to seed self.np_random
    super().reset(seed=seed)
    if seed is not None:
      self.locks[0].reset(seed)
      self.locks[1].reset(seed)
      if seed % 2 == 0:
        self.locks[0].optimal_reward = self.optimal_reward
        self.locks[1].optimal_reward = self.suboptimal_reward
      else:
        self.locks[0].optimal_reward = self.suboptimal_reward
        self.locks[1].optimal_reward = self.optimal_reward
    self.h = 0
    obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
    return obs, {}

  def step(self, action):
    assert self.n_locks == 2
    if self.h == 0:
      # In initial state, the action chooses lock
      self.lock_id = 0 if action < 5 else 1
      obs, _ = self.locks[self.lock_id].reset()
      reward, terminated = 0.0, False
      info = {'state': (0, self.h, self.lock_id)}
    else:
      obs, reward, terminated, _, info = self.locks[self.lock_id].step(action)
      info['state'] = info['state'] + (self.lock_id,)
    self.h += 1
    obs = np.array(np.append(obs, self.lock_id), dtype=np.uint8)
    return obs, reward, terminated, False, info

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
    self.action_space = spaces.Discrete(self.num_actions)
    # We encode the state type and time separately.
    # The time could be any value in 1 to horizon+1.
    # We further add noise of size horizon.
    self.obs_dim = 2 * horizon + 4
    self.observation_space = spaces.Box(low=0, high=1, shape=(self.obs_dim,), dtype=np.uint8)

  def _get_obs(self, s):
    if s is None or self.obs_dim is None:
      return s
    else:
      v = np.zeros(self.obs_dim)
      v[s[0]] = 1
      v[3 + s[1]] = 1
      return np.array(v, dtype=np.uint8)

  def _get_info(self, s):
    info = {"state": None if s is None else tuple(s)}
    return info

  def reset(self, seed=None, options=None):
    # We need the following line to seed self.np_random
    super().reset(seed=seed)
    if seed is not None:
      self.opt_a = self.np_random.integers(low=0, high=self.action_space.n, size=self.horizon, dtype=np.uint8)
      self.opt_b = self.np_random.integers(low=0, high=self.action_space.n, size=self.horizon, dtype=np.uint8)
    # Start stochastically in one of the two live states
    toss_value = self.np_random.binomial(1, 0.5)
    if toss_value == 0:
      self.state = np.array([0, 0], dtype=np.uint8)
    elif toss_value == 1:
      self.state = np.array([1, 0], dtype=np.uint8)
    else:
      raise AssertionError("Toss value can only be 1 or 0. Found %r" % toss_value)
    self.h = 0
    return self._get_obs(self.state), self._get_info(self.state)

  def transition(self, x, a):
    if x is None:
      raise Exception("Not in any state")
    b = self.np_random.binomial(1, self.swap)
    if x[0] == 0 and a == self.opt_a[x[1]]:
      if b == 0:
        return np.array([0, x[1]+1], dtype=np.uint8)
      else:
        return np.array([1, x[1]+1], dtype=np.uint8)
    if x[0] == 1 and a == self.opt_b[x[1]]:
      if b == 0:
        return np.array([1, x[1]+1], dtype=np.uint8)
      else:
        return np.array([0, x[1]+1], dtype=np.uint8)
    else:
      return np.array([2, x[1]+1], dtype=np.uint8)

  def reward(self, x, a, next_x):
    # If the agent reaches the final live states then give it the optimal reward.
    if (x[0] == 0 and x[1] == self.horizon-1 and a == self.opt_a[x[1]]) or (x[0] == 1 and x[1] == self.horizon-1 and a == self.opt_b[x[1]]):
      return self.optimal_reward * self.np_random.binomial(1, self.optimal_reward_prob)
    # If reaching the dead state for the first time then give it a small anti-shaping reward.
    # This anti-shaping reward is anti-correlated with the optimal reward.
    if x is not None and next_x is not None:
      if x[0] != 2 and next_x[0] == 2:
        return self.anti_shaping_reward * self.np_random.binomial(1, 0.5)
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
    obs = self._get_obs(self.state)
    terminated = self.h == self.horizon
    info = self._get_info(self.state)
    return obs, float(reward), terminated, False, info

  def render(self, mode="human"):
    return self._get_obs(self.state)
  
  def close(self):
    return None


if __name__ == '__main__':
  seed = 4
  env = DiabolicalCombLockEnv()
  env_cfg = {"horizon":5, "swap":0.5}
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