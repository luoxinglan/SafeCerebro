''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-04-01 14:29:59
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import numpy as np
import matplotlib.pyplot as plt
import joblib
data = joblib.load('./log/exp/exp_behavior_ddpg_seed_0/training_results/results.pkl')

episode = data['episode']
reward = list(map(lambda x: -x, data['episode_reward']))
# ego_records = data['ego_records']

plt.plot(episode, reward)
plt.xlabel('Episode')
plt.ylabel('Episode Reward')
plt.grid()
plt.xlim([0, 2000])
plt.tight_layout()
# plt.savefig('reward.png', dpi=300)
plt.show()