import math

import numpy as np
from gymnasium.envs.classic_control.mountain_car import MountainCarEnv


class SparseMountainCarEnv(MountainCarEnv):
  ''' Modified based on Mountain Car.
    The only difference is the reward function: 
    the agent gets 0 reward every step until it reaches the goal with 1 reward.
  '''
  def step(self, action: int):
    assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"

    position, velocity = self.state
    velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
    velocity = np.clip(velocity, -self.max_speed, self.max_speed)
    position += velocity
    position = np.clip(position, self.min_position, self.max_position)
    if position == self.min_position and velocity < 0:
      velocity = 0

    terminated = bool(
      position >= self.goal_position and velocity >= self.goal_velocity
    )
    # Set a sparse reward signal
    if terminated:
      reward = 1.0
    else:
      reward = 0.0

    self.state = (position, velocity)
    if self.render_mode == "human":
      self.render()
    # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
    return np.array(self.state, dtype=np.float32), reward, terminated, False, {}
  

if __name__ == '__main__':
  env = SparseMountainCarEnv()
  print('Action space:', env.action_space)
  print('Obsevation space:', env.observation_space)
  print('Obsevation space high:', env.observation_space.high)
  print('Obsevation space low:', env.observation_space.low)

  for i in range(1):
    obs, _ = env.reset(seed=i)
    env.action_space.seed(i)
    env.observation_space.seed(i)
    for _ in range(10):
      action = env.action_space.sample()
      obs, reward, terminated, _, _ = env.step(action)
      print('Observation:', obs)
      print('Reward:', reward)
      print('Done:', terminated)
      if terminated:
        break
  env.close()