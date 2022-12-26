from policy.lidar_dqn import Lidar_DQN
from policy.lidar_dqn_simple import Lidar_DQN_Simple
from policy.orca import ORCA
from policy.sac import SAC
from policy.ddpg import DDPG
from policy.sac_simple import SAC_Simple

policy_factory = dict()
policy_factory['lidar_dqn'] = Lidar_DQN
policy_factory['lidar_dqn_simple'] = Lidar_DQN_Simple
policy_factory['orca'] = ORCA
policy_factory['sac'] = SAC
policy_factory['ddpg'] = DDPG
policy_factory['sac_simple'] = SAC_Simple

