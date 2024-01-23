import gymnasium as gym
import gym_minatar
import gym_pygame
import gym_exploration

class RandomAgent(object):
  def __init__(self, action_space):
    self.action_space = action_space

  def act(self, observation, reward, terminated):
    return self.action_space.sample()

if __name__ == '__main__':
  seed = 42

  game = 'Catcher-PLE-v0'
  game = 'FlappyBird-PLE-v0'
  game = 'Pixelcopter-PLE-v0'
  game = 'PuckWorld-PLE-v0'
  game = 'Pong-PLE-v0'

  game = 'Asterix-MinAtar-v1'
  game = 'Breakout-MinAtar-v1'
  game = 'Freeway-MinAtar-v1'
  game = 'Seaquest-MinAtar-v1'
  game = 'SpaceInvaders-MinAtar-v1'

  game = 'NChain-v1'
  game = 'LockBernoulli-v0'
  game = 'LockGaussian-v0'
  game = 'SparseMountainCar-v0'
  game = 'DiabolicalCombLock-v0'

  env = gym.make(game)  
  if game in ['NChain-v1', 'LockBernoulli-v0', 'LockGaussian-v0', 'DiabolicalCombLock-v0']:
    game_cfg = {
      'NChain-v1': {'n':5},
      'LockBernoulli-v0': {'horizon':10, 'dimension':10, 'switch':0.1},
      'LockGaussian-v0': {'horizon':9, 'dimension':9, 'switch':0.1, 'noise':0.1},
      'DiabolicalCombLock-v0': {"horizon":5, "swap":0.5}
    }
    env.init(**game_cfg[game])

  obs, info = env.reset(seed=seed)
  env.action_space.seed(seed)
  env.observation_space.seed(seed)

  print('Game:', game)
  print('Action space:', env.action_space)
  print('Obsevation space:', env.observation_space)
  print('Obsevation space high:', env.observation_space.high)
  print('Obsevation space low:', env.observation_space.low)

  for _ in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, _, _ = env.step(action)
    # env.render() # default render mode is 'human'
    # env.render('human')
    # img = env.render('rgb_array')
    print('Observation:', obs)
    print('Reward:', reward)
    print('Done:', terminated)
    if terminated:
      break
  env.close()