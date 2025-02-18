''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-04-03 22:17:07
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import os
import numpy as np

import torch
import torch.nn as nn
from fnmatch import fnmatch
import torch.nn.functional as F
import torch.optim as optim

from safebench.util.torch_util import CUDA, CPU, hidden_init
from safebench.agent.base_policy import BasePolicy

class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        x = F.relu(self.fc2(x))
        return self.fc_out(x)
    
class DQN(BasePolicy):
    name = 'DQN'
    type = 'offpolicy'

    def __init__(self, config, logger):
        self.logger = logger
        self.continue_episode = 0
        self.action_dim = config['scenario_action_dim']
        self.state_dim = config['scenario_state_dim']
        self.hidden_dim = config['hidden_dim']
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.q_net = Qnet(self.state_dim, self.hidden_dim,self.action_dim).to(self.device)  # Q网络
        self.learning_rate = config['lr']
        self.gamma = config['gamma']
        self.epsilon = config['epsilon']
        self.target_update = config['target_update']
        self.buffer_start_training = config['buffer_start_training']
        self.batch_size = config['batch_size']
        self.buffer_capacity = config['buffer_capacity']
        
        # 目标网络
        self.target_q_net = Qnet(self.state_dim, self.hidden_dim, self.action_dim).to(self.device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr = self.learning_rate)
        self.count = 0  # 计数器,记录更新次数
        self.model_id = config['model_id']
        self.model_path = os.path.join(config['ROOT_DIR'], config['model_path'])
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.action_exp_spd_dlt = 0

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'train':
            self.q_net.train()
            self.target_q_net.train()
        elif mode == 'eval':
            self.q_net.eval()
            self.target_q_net.eval()
        else:
            raise ValueError(f'Unknown mode {mode}')
    

    def get_init_action(self, state, deterministic=False):
        num_scenario = len(state)
        additional_in = {}
        return [None] * num_scenario, additional_in

    def info_process(self, infos):
        info_batch = np.stack([i_i['actor_info'] for i_i in infos], axis=0)
        info_batch = info_batch.reshape(info_batch.shape[0], -1)
        return info_batch
    
    def get_action(self, state, infos, deterministic=False):
        if np.random.randn() > self.epsilon or deterministic: 
            state = self.info_process(infos)
            state = CUDA(torch.FloatTensor(state))
            self.action = self.q_net(state).argmax().item()
        else: # greedy policy
            self.action = np.random.randint(self.action_dim)
        if self.action == 0:
            self.action_exp_spd_dlt -= 0.1
        elif self.action == 1:
            self.action_exp_spd_dlt += 0.1
        else:
            raise ValueError(f'Unknown action {self.action}')
        self.action_exp_spd_dlt = min(max(self.action_exp_spd_dlt,-1),1)
        return [[self.action_exp_spd_dlt]]

    def train(self, replay_buffer):
        # check if memory is enough for one batch
        if replay_buffer.buffer_len < self.buffer_start_training:
            return

        # sample replay buffer
        batch = replay_buffer.sample(self.batch_size)
        bn_s = CUDA(torch.FloatTensor(batch['actor_info'])).reshape(self.batch_size, -1)
        bn_s_ = CUDA(torch.FloatTensor(batch['n_actor_info'])).reshape(self.batch_size, -1)
        bn_a = CUDA(torch.LongTensor(batch['action'])).unsqueeze(-1)
        bn_r = CUDA(-torch.FloatTensor(batch['reward'])).unsqueeze(-1) # [B, 1]
        bn_d = CUDA(torch.FloatTensor(batch['done'])).unsqueeze(-1) # [B, 1]
            
        q_values = self.q_net(bn_s).gather(1, bn_a)  # Q值
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(bn_s_).max(1)[0].view(-1, 1)
        q_targets = bn_r + self.gamma * max_next_q_values * (1 - bn_d)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络
        self.count += 1

    def save_model(self, episode):
        states = {
            'q_net': self.q_net.state_dict(),
            'target_q_net': self.target_q_net.state_dict(),
        }
        filepath = os.path.join(self.model_path, f'model.dqn.{self.model_id}.{episode:04}.torch')
        self.logger.log(f'>> Saving scenario policy {self.name} model to {filepath}')
        with open(filepath, 'wb+') as f:
            torch.save(states, f)

    def load_model(self, episode=None):
        if episode is None:
            episode = -1
            for _, _, files in os.walk(self.model_path):
                for name in files:
                    if fnmatch(name, "*torch"):
                        cur_episode = int(name.split(".")[-2])
                        if cur_episode > episode:
                            episode = cur_episode
        filepath = os.path.join(self.model_path, f'model.dqn.{self.model_id}.{episode:04}.torch')
        if os.path.isfile(filepath):
            self.logger.log(f'>> Loading scenario policy {self.name} model from {filepath}')
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)
            self.q_net.load_state_dict(checkpoint['q_net'])
            self.target_q_net.load_state_dict(checkpoint['target_q_net'])
            self.continue_episode = episode
        else:
            self.logger.log(f'>> No scenario policy {self.name} model found at {filepath}', 'red')
            # exit()