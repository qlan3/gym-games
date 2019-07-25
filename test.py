import gym
import gym_minatar
import gym_pygame

class RandomAgent(object):
  def __init__(self, action_space):
    self.action_space = action_space

  def act(self, observation, reward, done):
    return self.action_space.sample()

if __name__ == '__main__':
  game = 'Catcher-PLE-v0'
  game = 'FlappyBird-PLE-v0'
  game = 'Pixelcopter-PLE-v0'
  game = 'PuckWorld-PLE-v0'
  game = 'Pong-PLE-v0'
  
  game = 'Asterix-MinAtar-v0'
  game = 'Breakout-MinAtar-v0'
  game = 'Freeway-MinAtar-v0'
  game = 'Seaquest-MinAtar-v0'
  game = 'Space_invaders-MinAtar-v0'

  env = gym.make(game)
  env.seed(0)

  print('Action space:', env.action_space)
  print('Obsevation space:', env.observation_space)
  print('Obsevation space high:', env.observation_space.high)
  print('Obsevation space low:', env.observation_space.low)

  for i in range(1):
    ob = env.reset()
    for _ in range(10):
      action = env.action_space.sample()
      ob, reward, done, _ = env.step(action)
      env.render() # default render mode is 'human'
      # env.render('human')
      # img = env.render('rgb_array')
      print('Observation:', ob)
      print('Reward:', reward)
      print('Done:', done)
      if done:
        break
  env.close()