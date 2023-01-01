from gym.envs.registration import register

for game in ['LockBernoulli', 'LockGaussian', 'SparseMountainCar', 'DiabolicalCombLock']:
  register(
    id='{}-v0'.format(game),
    entry_point=f'gym_exploration.envs:{game}Env'
  )

register(
  id='{}-v1'.format('NChain'),
  entry_point=f'gym_exploration.envs:NChainEnv'
)