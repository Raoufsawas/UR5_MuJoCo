import gym


from stable_baselines import ACER, DQN , PPO2, A2C, ACKTR
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.cmd_util import make_vec_env
import numpy as np
from stable_baselines import DQN
import os
import time
from gym_ur5.controller.MujocoController import MJ_Controller


#Training
#env = make_vec_env('gym_ur5:UR5-v0', n_envs=2)

#model = ACER(MlpPolicy, env, verbose=1, tensorboard_log="./logs/")
#model.learn(total_timesteps=500000)
#model.save("ACER")

################################################################
# Test the trained agent
env = make_vec_env('gym_ur5:UR5-v0', n_envs=1)

model = ACER(MlpPolicy, env, verbose=1)

#model = A2C.load("A2C")
model = ACER.load("ACER")


obs = env.reset()
n_steps = 10000
for step in range(n_steps):
  action, _ = model.predict(obs, deterministic=True)
  print("Step {}".format(step + 1))
  print("Action: ", action)
  obs, reward, done, info = env.step(action)
  print('obs=', obs, 'reward=', reward, 'done=', done)
