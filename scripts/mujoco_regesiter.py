print(__name__)
import gym
from dmbrl import env

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