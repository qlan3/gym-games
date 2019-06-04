import gym
import gym_pygame

class RandomAgent(object):
  def __init__(self, action_space):
    self.action_space = action_space

  def act(self, observation, reward, done):
    return self.action_space.sample()

if __name__ == '__main__':
  game = 'Catcher-v0'
  game = 'FlappyBird-v0'
  game = 'Pixelcopter-v0'
  game = 'PuckWorld-v0'
  game = 'PongPLE-v0'

  env = gym.make(game)
  env.seed(0)

  print('Action space:', env.action_space)
  print('Obsevation space:', env.observation_space)
  print('Obsevation space high:', env.observation_space.high)
  print('Obsevation space low:', env.observation_space.low)

  for i in range(1):
    ob = env.reset()
    for _ in range(1000):
      action = env.action_space.sample()
      ob, reward, done, _ = env.step(action)
      env.render()
      print('Observation:', ob)
      print('Reward:', reward)
      print('Done:', done)
      if done:
        break
  env.close()