from gym.envs.registration import register


for game in ["asterix", "breakout", "freeway", "seaquest", "space_invaders"]:
  name = game.title().replace('_', '')
  register(
    id='{}-MinAtar-v0'.format(name),
    entry_point=f'gym_minatar.envs:BaseEnv',
    kwargs=dict(game=game, display_time=50, use_minimal_action_set=False),
  )
  register(
    id='{}-MinAtar-v1'.format(name),
    entry_point=f'gym_minatar.envs:BaseEnv',
    kwargs=dict(game=game, display_time=50, use_minimal_action_set=True),
  )