from gym_pygame.envs.base import BaseEnv


class FlappyBirdEnv(BaseEnv):
  def __init__(self, normalize=False, display=False, **kwargs):
    self.game_name = 'FlappyBird'
    self.init(normalize, display, **kwargs)
    
  def get_ob_normalize(cls, state):
    state_normal = cls.get_ob(state)
    # TODO
    return state_normal

if __name__ == '__main__':
  env = FlappyBirdEnv(normalize=True)
  print('Action space:', env.action_space)
  print('Action set:', env.action_set)
  print('Obsevation space:', env.observation_space)
  print('Obsevation space high:', env.observation_space.high)
  print('Obsevation space low:', env.observation_space.low)

  seed = 42
  obs, _ = env.reset(seed)
  obs, info = env.reset(seed)
  env.action_space.seed(seed)
  env.observation_space.seed(seed)
  while True:
    action = env.action_space.sample()
    obs, reward, terminated, _, _ = env.step(action)
    # env.render('rgb_array')
    # env.render('human')
    print('Observation:', obs)
    print('Reward:', reward)
    print('Done:', terminated)
    if terminated:
      # obs, _ = env.reset()
      break
  env.close()