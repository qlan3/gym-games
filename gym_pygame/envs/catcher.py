import os
import importlib
import numpy as np
import gym
from gym import spaces
from gym.envs.classic_control import rendering
from ple import PLE


class CatcherEnv(gym.Env):
  metadata = {'render.modes': ['human', 'rgb_array']}
  
  @classmethod
  def get_ob(cls, state):
    return np.array([state['player_x'], state['player_vel'], state['fruit_x'], state['fruit_y']])
  
  @classmethod
  def get_ob_normalize(cls, state):
    state_normal = cls.get_ob(state)
    state_normal[0] = (state_normal[0] - 26) / 26
    state_normal[1] = (state_normal[1]) / 8
    state_normal[2] = (state_normal[2] - 26) / 26
    state_normal[3] = (state_normal[3] - 20) / 45
    return state_normal

  def __init__(self, normalize=True, display=False, **kwargs):
    game_name = 'Catcher'
    game_module_name = f'ple.games.{game_name.lower()}'
    game_module = importlib.import_module(game_module_name)
    self.game = getattr(game_module, game_name)(**kwargs)
    
    if display==False:
      # Do not open a PyGame window
      os.putenv('SDL_VIDEODRIVER', 'fbcon')
      os.environ["SDL_VIDEODRIVER"] = "dummy"

    if normalize:
        self.gameOb = PLE(self.game, fps=30, state_preprocessor=self.get_ob_normalize, display_screen=display)
    else:
        self.gameOb = PLE(self.game, fps=30, state_preprocessor=self.get_ob, display_screen=display)
    
    self.viewer = None
    self.action_set = self.gameOb.getActionSet()
    self.action_space = spaces.Discrete(len(self.action_set))
    self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(self.game.getGameState()),), dtype=np.float32)
    self.gameOb.init()

  def step(self, action):
    reward = self.gameOb.act(self.action_set[action])
    done = self.gameOb.game_over()
    return (self.gameOb.getGameState(), reward, done, {})
    
  def reset(self):
    self.gameOb.reset_game()
    return self.gameOb.getGameState()
  
  def seed(self, seed=None):
    self.gameOb.rng.seed(seed)
    self.gameOb.init()
    return seed

  def render(self, mode='human'):
    # img = self.gameOb.getScreenRGB() 
    # img = self.gameOb.getScreenGrayscale()
    img = np.fliplr(np.rot90(self.gameOb.getScreenRGB(),3))
    if mode == 'rgb_array':
      # Return a np array with shape = [64, 64, 3] 
      assert img.shape == [64, 64, 3]
      return img
    elif mode == 'human':
      if self.viewer is None:
        self.viewer = rendering.SimpleImageViewer()
      self.viewer.imshow(img)

  def close(self):
    if self.viewer != None:
      self.viewer.close()
      self.viewer = None
    return 0

if __name__ == '__main__':
  env = CatcherEnv()
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
      env.render()
      print('Observation:', ob)
      print('Reward:', reward)
      print('Done:', done)
      if done:
        break
  env.close()