from gym.envs.registration import register
from gym_ur5.version import VERSION as __version__

reg=register(
id='UR5-v0',
entry_point='gym_ur5.envs:UR5Env',)
