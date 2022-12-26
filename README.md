# RLDynamicNav
Video Link: https://youtu.be/f2ktS3TSF-g

# Environment
- tensorflow-gpu >= 2.4
- Cython
- ORCA library, https://github.com/sybrenstuvel/Python-RVO2

# Compile the LiDAR scan library
in folder DRL_dynamic_navigation/C_library, ```python setup.py build_ext --inplace```

# select algorithm
- in file DRL_dynamic_navigation/train.py
revise this line: parser.add_argument('--policy', type=str, default='lidar_dqn')
- lidar_dqn: original method with DQN RL
- lidar_dqn_simple: without CNN
- sac: replace DQN with SAC
- sac_simple: SAC without CNN
- ddpg: DDPG without CNN

# select reward type
parser.add_argument('--reward_simple', default=False, action='store_true')
- False, original reward
- True, without distance reward

# select environment type
parser.add_argument('--complex_environment', default=True, action='store_true')
- False, original environment only with dynamic obstacles
- True, with static obstacles
