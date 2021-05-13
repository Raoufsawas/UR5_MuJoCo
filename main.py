import gym


from stable_baselines import ACER, DQN , PPO2, A2C, ACKTR
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.cmd_util import make_vec_env
import numpy as np
#from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
#from stable_baselines.common.vec_env import DummyVecEnv
#from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN
import os
import time
from gym_ur5.controller.MujocoController import MJ_Controller


env = make_vec_env('gym_ur5:UR5-v0', n_envs=1)

model = ACER(MlpPolicy, env, verbose=1, tensorboard_log="./a2c_cartpole_tensorboard/")
#model.learn(total_timesteps=500000)
#model.save("ACERnew1")

#env = gym.make('gym_ur5:UR5-v0')
#env = make_vec_env('gym_ur5:UR5-v0', n_envs=1)
#action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
#env = VecNormalize(env,norm_obs=True)

#model = DQN(MlpPolicy, env, verbose=1, tensorboard_log="./a2c_cartpole_tensorboard/")
# wrap it
#env = make_vec_env(lambda: env, n_envs=4)
# Train the agent
#model = A2C(MlpPolicy, env, verbose=1,  tensorboard_log="./a2c_cartpole_tensorboard/")
#model.learn(total_timesteps=500000)
#del model
#model.save("A2C_new41")
#model = A2C.load("Ur5A2C")
model = ACER.load("ACERnew")
#model.learn(total_timesteps=5000)
# Test the trained agent
obs = env.reset()
n_steps = 10000
for step in range(n_steps):
  action, _ = model.predict(obs, deterministic=True)
  #print("Step {}".format(step + 1))
  #print("Action: ", action)
  obs, reward, done, info = env.step(action)
  #print('obs=', obs, 'reward=', reward, 'done=', done)
