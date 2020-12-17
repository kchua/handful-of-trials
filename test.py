from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from matplotlib import pyplot as plt
import sys


# def shuffle_rows(arr):
#     idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
#     return arr[np.arange(arr.shape[0])[:, None], idxs]

# a = np.random.uniform(size=(3,3,3))
# print(a)
# a = shuffle_rows(a)
# print(a)


import gym
from dmbrl.misc.DotmapUtils import get_required_argument
from dmbrl.modeling.layers import FC
import dmbrl.env
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

gym.spaces
print(__name__)
env = gym.make("MBRLHalfCheetah-v0")
observation = env.reset()
for step in range(1000):
  # env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)
  print(step)
  if done:
    observation = env.reset()
env.close()