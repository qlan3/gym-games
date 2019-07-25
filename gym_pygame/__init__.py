from gym.envs.registration import register

for game in ['Catcher', 'FlappyBird', 'Pixelcopter', 'PuckWorld', 'Pong']:
  register(
    id='{}-PLE-v0'.format(game),
    entry_point=f'gym_pygame.envs:{game}Env'
)