import numpy as np
import gym

def make_env(env_name, seed=-1, render_mode=False):
  if (env_name.startswith("RacecarBulletEnv")):
    import pybullet as p
    import pybullet_envs
    import pybullet_envs.bullet.racecarGymEnv as racecarGymEnv
    print("bullet_racecar_started")
    env = racecarGymEnv.RacecarGymEnv(isDiscrete=False, renders=render_mode)
  elif (env_name.startswith("RocketLander")):
    from box2d.rocket import RocketLander
    env = RocketLander()
  elif (env_name.startswith("BipedalWalker")):
    if (env_name.startswith("BipedalWalkerHardcore")):
      from box2d.biped import BipedalWalkerHardcore
      env = BipedalWalkerHardcore()
    else:
      from box2d.biped import BipedalWalker
      env = BipedalWalker()
  elif (env_name.startswith("MinitaurBulletEnv")):
    import pybullet as p
    import pybullet_envs
    import pybullet_envs.bullet.minitaur_gym_env as minitaur_gym_env
    print("bullet_minitaur_started")
    env = minitaur_gym_env.MinitaurBulletEnv(render=render_mode)
  elif (env_name.startswith("MinitaurDuckBulletEnv")):
    print("bullet_minitaur_duck_started")
    import pybullet as p
    import pybullet_envs
    from custom_envs.minitaur_duck import MinitaurDuckBulletEnv
    env = MinitaurDuckBulletEnv(render=render_mode)
  elif (env_name.startswith("MinitaurBallBulletEnv")):
    print("bullet_minitaur_ball_started")
    import pybullet as p
    import pybullet_envs
    from custom_envs.minitaur_ball import MinitaurBallBulletEnv
    env = MinitaurBallBulletEnv(render=render_mode)
  elif (env_name.startswith("SlimeVolley")):
    print("slimevolley_swingup_started")
    from custom_envs.slimevolley import SlimeVolleyEnv, SurvivalRewardEnv
    env = SlimeVolleyEnv()
    env = SurvivalRewardEnv(env) # optional
  elif (env_name.startswith("CartPoleSwingUp")):
    print("cartpole_swingup_started")
    from custom_envs.cartpole_swingup import CartPoleSwingUpEnv
    env = CartPoleSwingUpEnv()
  elif (env_name.startswith("KukaBulletEnv")):
    import pybullet as p
    import pybullet_envs
    import pybullet_envs.bullet.kukaGymEnv as kukaGymEnv
    print("bullet_kuka_grasping started")
    env = kukaGymEnv.KukaGymEnv(renders=render_mode,isDiscrete=False)
  else:
    if env_name.startswith("Roboschool"):
      import roboschool
    env = gym.make(env_name)
    if render_mode and not env_name.startswith("Roboschool"):
      env.render("human")
  if (seed >= 0):
    env.seed(seed)
  '''
  print("environment details")
  print("env.action_space", env.action_space)
  print("high, low", env.action_space.high, env.action_space.low)
  print("environment details")
  print("env.observation_space", env.observation_space)
  print("high, low", env.observation_space.high, env.observation_space.low)
  assert False
  '''
  return env
