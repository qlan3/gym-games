from gym.envs.registration import register

for game in ['Catcher']:
  register(
    id='{}-v0'.format(game),
    entry_point=f'gym_pygame.envs:{game}Env'
)