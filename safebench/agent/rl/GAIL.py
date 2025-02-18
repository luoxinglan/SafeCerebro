''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-04-01 15:59:51
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
from torch.distributions import Normal
import torch.optim as optim

from safebench.util.torch_util import CUDA, CPU, hidden_init
from safebench.agent.base_policy import BasePolicy
from safebench.scenario.scenario_policy.rl.utils import rl_utils_ppo

class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, int(hidden_dim/2))
        self.fc_mu = torch.nn.Linear(int(hidden_dim/2), action_dim)
        self.fc_std = torch.nn.Linear(int(hidden_dim/2), action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = 1 * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x)) + 1e-8
        return mu, std

class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, int(hidden_dim/2))
        self.fc3 = torch.nn.Linear(int(hidden_dim/2), 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, policy_lr, value_lr,
                 lmbda, epochs, eps, gamma, device):
        self.policy = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.value = ValueNet(state_dim, hidden_dim).to(device)
        self.optim = torch.optim.Adam(self.policy.parameters(),lr=policy_lr)
        self.value_optim = torch.optim.Adam(self.value.parameters(),lr=value_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用于训练轮数
        self.clip_epsilon = eps  # PPO中截断范围的参数
        self.device = device

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'train':
            self.policy.train()
            self.value.train()
        elif mode == 'eval':
            self.policy.eval()
            self.value.eval()
        else:
            raise ValueError(f'Unknown mode {mode}')
    
    def take_action(self, state, infos, deterministic=False):
        state_tensor = CUDA(torch.FloatTensor(state))
        mu, sigma = self.policy(state_tensor)
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        return action.cpu().numpy()

    def update(self, transition_dict):
        bn_s = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        bn_a = torch.tensor(transition_dict['actions']).to(
            self.device)
        bn_r = torch.tensor(transition_dict['rewards'],
                            dtype=torch.float).to(self.device)
        bn_s_ = torch.tensor(transition_dict['next_states'],
                                dtype=torch.float).to(self.device)
        bn_d = torch.tensor(transition_dict['dones'],
                            dtype=torch.float).to(self.device).unsqueeze(1)
        
        td_target = bn_r + self.gamma * self.value(bn_s_) * (1 - bn_d)
        td_delta = td_target - self.value(bn_s)
        advantage = rl_utils_ppo.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        mu, std = self.policy(bn_s)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        old_log_probs = action_dists.log_prob(bn_a)

        for K in range(self.epochs):
            # ---------------------------------------
            mu, std = self.policy(bn_s)
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(bn_a)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage
            policy_loss = torch.mean(-torch.min(surr1, surr2))
            value_loss = torch.mean(F.mse_loss(self.value(bn_s), td_target.detach()))
            self.optim.zero_grad()
            self.value_optim.zero_grad()
            policy_loss.backward()
            value_loss.backward()
            self.optim.step()
            self.value_optim.step()
            # -------------------------------------------


class Discriminator(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Discriminator, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        return torch.sigmoid(self.fc2(x))

class GAIL(BasePolicy):
    name = 'GAIL'
    type = 'IRL'

    def __init__(self, config, logger):
        self.logger = logger

        self.continue_episode = 0
        self.state_dim = config['ego_state_dim']
        self.hidden_dim = config['hidden_dim']
        self.action_dim = config['ego_action_dim']
        self.lr_d = config['lr_d']
        self.policy_lr = config['policy_lr']
        self.value_lr = config['value_lr']
        self.lmbda = config['lmbda']
        self.batch_size = config['batch_size']
        self.train_iteration = config['train_iteration']
        self.clip_epsilon = config['clip_epsilon']
        self.gamma = config['gamma']
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        torch.manual_seed(0)
        self.lambda_ = 0.1 #奖励约束
        self.lambda__ = 0.2 #形状约束
        
        self.model_id = config['model_id']
        self.model_path = os.path.join(config['ROOT_DIR'], config['model_path'])
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.discriminator = Discriminator(self.state_dim, self.hidden_dim, self.action_dim).to(self.device)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr_d)
        
        # load expert data
        data_expert_ = np.load('./log/exp/exp_basic_standard_seed_0/eval_results/trajectory_expert.npy',allow_pickle=True)
        # 初始化空列表来收集每个部分的数据
        s_expert_ = []
        a_expert_ = []
        for data_dict_ in data_expert_:
            s_expert_.append(data_dict_['obs:'])
            a_expert_.append(data_dict_['ego_actions:'])
        # 将列表转换为numpy数组
        self.state_expert = np.concatenate(s_expert_, axis=0)
        self.action_expert = np.concatenate(a_expert_, axis=0)

        self.agent = PPO(self.state_dim, self.hidden_dim, self.action_dim, self.policy_lr, self.value_lr,
                 self.lmbda, self.train_iteration, self.clip_epsilon, self.gamma, self.device) 

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'train':
            self.discriminator.train()
            self.agent.set_mode('train')
        elif mode == 'eval':
            self.discriminator.eval()
            self.agent.set_mode('eval')
        else:
            raise ValueError(f'Unknown mode {mode}')

    def get_action(self, state, infos, deterministic=False):
        action = self.agent.take_action(state, infos, deterministic=False)
        return action
    
    def compute_state_transition_similarity(self, agent_next_states, expert_next_states):
        """
        计算代理状态转移与专家状态转移的相似度。
        这里使用欧氏距离作为相似度的度量，也可以根据需要选择其他度量方式。
        """
        # 计算代理状态转移与每个专家状态转移的欧氏距离
        distances = torch.cdist(agent_next_states, expert_next_states, p=2)
        # 取最小距离作为相似度度量，这里假设距离越小，相似度越高
        min_distances, _ = distances.min(dim=1)
        # 将距离转换为奖励，这里简单地取负值，也可以根据需要设计更复杂的转换函数
        similarity_rewards = -min_distances
        return similarity_rewards
    
    def learn(self, buffer_record):
        state_list = []
        action_list = []
        next_state_list = []
        rewards_list = []
        done_list = []
        for state, action, next_state, rewards ,done in buffer_record:
            state_list.append(state[0])
            action_list.append(action[0])
            next_state_list.append(next_state[0])
            rewards_list.append(rewards[0])
            done_list.append(done[0])

        expert_states = torch.tensor(self.state_expert, dtype=torch.float).to(self.device)
        expert_actions = torch.tensor(self.action_expert, dtype=torch.float).to(self.device)
        agent_states = torch.tensor(state_list, dtype=torch.float).to(self.device)
        agent_actions = torch.tensor(action_list, dtype=torch.float).to(self.device) #离散数据还需要独热编码one_hot

        expert_prob = self.discriminator(expert_states, expert_actions)
        agent_prob = self.discriminator(agent_states, agent_actions)
        discriminator_loss = nn.BCELoss()(agent_prob, torch.ones_like(agent_prob)) + nn.BCELoss()(expert_prob, torch.zeros_like(expert_prob))
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()
        print('discriminator_loss:',discriminator_loss.cpu().item())
        
        agent_next_states = torch.tensor(next_state_list, dtype=torch.float).to(self.device)
        expert_next_states = torch.tensor(self.state_expert, dtype=torch.float).to(self.device)
        state_transition_rewards = self.compute_state_transition_similarity(agent_next_states, expert_next_states)
    
        rewards = -torch.log(agent_prob).detach().cpu().numpy()+self.lambda_*np.array(rewards_list)[:,np.newaxis]\
            +self.lambda__*state_transition_rewards.unsqueeze(1).cpu().numpy()

        transition_dict = {
            'states': state_list,
            'actions': action_list,
            'rewards': rewards,
            'next_states': next_state_list,
            'dones': done_list
        }
        self.agent.update(transition_dict) #每一次transition_dict需要清空
        

    def save_model(self, episode):
        states = {
            'discriminator': self.discriminator.state_dict(),
            'policy': self.agent.policy.state_dict(),
            'value': self.agent.value.state_dict(),
        }
        filepath = os.path.join(self.model_path, f'model.GAIL.{self.model_id}.{episode:04}.torch')
        self.logger.log(f'>> Saving {self.name} model to {filepath}')
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
        filepath = os.path.join(self.model_path, f'model.GAIL.{self.model_id}.{episode:04}.torch')
        if os.path.isfile(filepath):
            self.logger.log(f'>> Loading {self.name} model from {filepath}')
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)
            self.discriminator.load_state_dict(checkpoint['discriminator'])
            self.agent.policy.load_state_dict(checkpoint['policy'])
            self.agent.value.load_state_dict(checkpoint['value'])
            self.continue_episode = episode
        else:
            self.logger.log(f'>> No {self.name} model found at {filepath}', 'red')
