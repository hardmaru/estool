import numpy as np
import gym

# change the reward for distance travelled (in the right direction) only.
DISTANCE_REWARD_MODE = True

class NoDeath(gym.Wrapper):
  def __init__(self, env, t_max=1000):
    """
    Replace death event with a reward of -1.
    But will die after t_max.
    """
    gym.Wrapper.__init__(self, env)
    self.t_max = t_max
    self.t = 0

  def reset(self):
    obs = self.env.reset()
    self.t = 0
    return obs

  def step(self, action):
    self.t += 1
    obs, reward, virtual_done, info = self.env.step(action)
    if DISTANCE_REWARD_MODE:
      reward = self.env.rewards[1] # progress (distance).
    done = False
    if self.t >= self.t_max:
      done = True
    if virtual_done:
      reward -= 1.0
    return obs, reward, done, info
