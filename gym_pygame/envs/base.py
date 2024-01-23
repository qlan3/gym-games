import os
import importlib
import numpy as np
from ple import PLE

import gymnasium as gym
from gymnasium import spaces


class BaseEnv(gym.Env):
  metadata = {"render_modes": ["human", "array", "rgb_array"]}

  def __init__(self, normalize=False, display=False, **kwargs):
    self.game_name = 'Game Name'
    self.init(normalize, display, **kwargs)
    
  def init(self, normalize, display, **kwargs):
    game_module_name = f'ple.games.{self.game_name.lower()}'
    game_module = importlib.import_module(game_module_name)
    self.game = getattr(game_module, self.game_name)(**kwargs)

    if display == False:
      # Do not open a PyGame window
      os.putenv('SDL_VIDEODRIVER', 'fbcon')
      os.environ['SDL_VIDEODRIVER'] = 'dummy'
    
    if normalize:
        self.gameOb = PLE(self.game, fps=30, state_preprocessor=self.get_ob_normalize, display_screen=display)
    else:
        self.gameOb = PLE(self.game, fps=30, state_preprocessor=self.get_ob, display_screen=display)
    
    self.viewer = None
    self.action_set = self.gameOb.getActionSet()
    self.action_space = spaces.Discrete(len(self.action_set))
    self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(self.game.getGameState()),), dtype=np.float32)
    self.gameOb.init()

  def get_ob(self, state):
    return np.array(list(state.values()), dtype=np.float32)

  def get_ob_normalize(self, state):
    raise NotImplementedError('Get observation normalize function is not implemented!')
    
  def reset(self, seed=None, options=None):
    if seed is not None:
      self.gameOb.rng.seed(seed)
      self.gameOb.init()
    self.gameOb.reset_game()
    return self.gameOb.getGameState(), {}

  def step(self, action):
    reward = self.gameOb.act(self.action_set[action])
    terminated = self.gameOb.game_over()
    return self.gameOb.getGameState(), reward, terminated, False, {}

  def render(self, mode='human'):
    # img = self.gameOb.getScreenRGB() 
    # img = self.gameOb.getScreenGrayscale()
    img = np.fliplr(np.rot90(self.gameOb.getScreenRGB(),3))
    if mode == 'rgb_array':
      return img
    elif mode == 'human':
      from gym_pygame.envs import rendering
      if self.viewer is None:
        self.viewer = rendering.SimpleImageViewer()
      self.viewer.imshow(img)

  def close(self):
    if self.viewer != None:
      self.viewer.close()
      self.viewer = None
    return 0