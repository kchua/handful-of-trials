import gym

env = gym.make("Ant-v1")
ac_ub, ac_lb = env.action_space.high, env.action_space.low
O, A = env.observation_space.shape[0], env.action_space.shape[0]
print(ac_ub.shape, ac_ub, ac_lb)
print(O, A)

# out put
# (6,) [1. 1. 1. 1. 1. 1.] [-1. -1. -1. -1. -1. -1.]
# 17, 6