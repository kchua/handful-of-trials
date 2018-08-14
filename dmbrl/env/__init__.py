from gym.envs.registration import register


register(
    id='MBRLCartpole-v0',
    entry_point='dmbrl.env.cartpole:CartpoleEnv'
)


register(
    id='MBRLReacher3D-v0',
    entry_point='dmbrl.env.reacher:Reacher3DEnv'
)


register(
    id='MBRLPusher-v0',
    entry_point='dmbrl.env.pusher:PusherEnv'
)


register(
    id='MBRLHalfCheetah-v0',
    entry_point='dmbrl.env.half_cheetah:HalfCheetahEnv'
)
