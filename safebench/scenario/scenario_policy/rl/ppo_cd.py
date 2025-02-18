''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-04-03 22:17:00
Description: 
    Copyright (c) 2022-2023 Safebench Team

    Modified from <https://github.com/gouxiangchen/ac-ppo>

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import os

import numpy as np
import torch
import torch.nn as nn
from fnmatch import fnmatch
from torch.distributions import Normal
import torch.nn.functional as F

from safebench.util.torch_util import CUDA, CPU, hidden_init
from safebench.agent.base_policy import BasePolicy
from safebench.scenario.scenario_policy.rl.utils import CuriosityDriven, rl_utils_ppo


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        hidden_dim = 64
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.ln1 = torch.nn.LayerNorm(hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = torch.nn.LayerNorm(hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc_mu.weight.data.uniform_(*hidden_init(self.fc_mu))
        self.fc_std.weight.data.uniform_(*hidden_init(self.fc_std))

    def forward(self, x):
        x = self.ln1(F.relu(self.fc1(x)))
        x = self.ln2(F.relu(self.fc2(x)))
        mu = 1 * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x)) + 1e-8
        return mu, std

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        hidden_dim = 64
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.ln1 = torch.nn.LayerNorm(hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = torch.nn.LayerNorm(hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))

    def forward(self, x):
        x = self.ln1(F.relu(self.fc1(x)))
        x = self.ln2(F.relu(self.fc2(x)))
        return self.fc3(x)

class PPO_CD(BasePolicy):
    name = 'PPO_CD'
    type = 'onpolicy'

    def __init__(self, config, logger):
        super(PPO_CD, self).__init__(config, logger)

        self.continue_episode = 0
        self.logger = logger
        self.gamma = config['gamma']
        self.policy_lr = config['policy_lr']
        self.value_lr = config['value_lr']
        self.train_iteration = config['train_iteration']
        self.state_dim = config['scenario_state_dim']
        self.action_dim = config['scenario_action_dim']
        self.clip_epsilon = config['clip_epsilon']
        self.batch_size = config['batch_size']
        self.lmbda = config['lmbda']
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.action_bound = 1

        self.model_id = config['model_id']
        self.model_path = os.path.join(config['ROOT_DIR'], config['model_path'])
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.policy = CUDA(PolicyNetwork(state_dim=self.state_dim, action_dim=self.action_dim))
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=self.policy_lr)
        self.value = CUDA(ValueNetwork(state_dim=self.state_dim))
        self.value_ins = CUDA(ValueNetwork(state_dim=self.state_dim))
        self.value_optim = torch.optim.Adam(self.value.parameters(), lr=self.value_lr)
        self.value_ins_optim = torch.optim.Adam(self.value_ins.parameters(), lr=self.value_lr)
        self.cd = CuriosityDriven.CD(self.state_dim, 32, 1, 1)
        torch.manual_seed(seed=10)
        self.mode = 'train'

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'train':
            self.policy.train()
            self.value.train()
            self.value_ins.train()
            self.cd.set_mode('train')
        elif mode == 'eval':
            self.policy.eval()
            self.value.eval()
            self.value_ins.eval()
            self.cd.set_mode('eval')
        else:
            raise ValueError(f'Unknown mode {mode}')

    def info_process(self, infos):
        info_batch = np.stack([i_i['actor_info'] for i_i in infos], axis=0)
        info_batch = info_batch.reshape(info_batch.shape[0], -1)
        return info_batch

    def get_init_action(self, state, deterministic=False):
        num_scenario = len(state)
        additional_in = {}
        return [None] * num_scenario, additional_in

    def get_action(self, state, infos, deterministic=False):
        state = self.info_process(infos)
        normalized_state_tensor = CUDA(torch.FloatTensor(state))
        mu, sigma = self.policy(normalized_state_tensor)
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        action.clamp_(-self.action_bound, self.action_bound)
        return [[action.item()]]
    
    def train(self, replay_buffer):
        batch_size = replay_buffer.buffer_len
        batch = replay_buffer.sample(batch_size, False)
        bn_s = CUDA(torch.FloatTensor(batch['actor_info'])).reshape(batch_size, -1)
        bn_s_ = CUDA(torch.FloatTensor(batch['n_actor_info'])).reshape(batch_size, -1)
        bn_a = CUDA(torch.FloatTensor(batch['action']))
        bn_r = CUDA(-torch.FloatTensor(batch['reward'])).unsqueeze(-1) # [B, 1]

        bn_d = CUDA(torch.FloatTensor(batch['done'])).unsqueeze(-1)

        rewards_attacker = bn_r
        # rewards_attacker = torch.where(bn_r > 1, torch.ones_like(bn_r), -torch.ones_like(bn_r))
        td_target = rewards_attacker + self.gamma * self.value(bn_s_)*(1-bn_d)
        td_delta = td_target - self.value(bn_s)
        advantage_attack = rl_utils_ppo.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        mu, std = self.policy(bn_s)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        old_log_probs = action_dists.log_prob(bn_a)


        # Curiosity-Driven ------------------------------------------------------------------
        # rewards_env = torch.where(bn_r > 1, -torch.ones_like(bn_r), torch.ones_like(bn_r)) #R_v=R_joint(s_t,a(a_attack,a_victim)),根据情况更改！
        rewards_env = -0*bn_r
        V_victim_cur, V_victim_next = self.cd.surrogate_update(bn_s,rewards_env,bn_s_,bn_d)
        reward_intrinsic = self.cd.RND_update()
        td_delta_victim = rewards_env+self.gamma*V_victim_next*(1-bn_d)-V_victim_cur
        advantage_victim = rl_utils_ppo.compute_advantage(self.gamma, self.lmbda, td_delta_victim.cpu()).to(self.device)
        td_delta_attack_ins = reward_intrinsic+self.gamma*self.value_ins(bn_s_)*(1-bn_d)-self.value_ins(bn_s)
        advantage_attack_ins = rl_utils_ppo.compute_advantage(self.gamma, self.lmbda, td_delta_attack_ins.cpu()).to(self.device)
        lambda_ = 0.2 #the degree of exploration
        advantage_attack = advantage_attack+lambda_*advantage_attack_ins
        #------------------------------------------------------------------------------------
        

        # start to train, use gradient descent without batch size
        for K in range(self.train_iteration):
            mu, std = self.policy(bn_s)
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(bn_a)
            ratio = torch.exp(log_probs - old_log_probs)

            # L1 = ratio * advantage_attack
            # L2 = torch.clamp(ratio, 1.0-self.clip_epsilon, 1.0+self.clip_epsilon) * advantage_attack
            # policy_loss = torch.mean(-torch.min(L1, L2))

            L1 = torch.min(torch.clamp(ratio, 1.0-self.clip_epsilon, 1.0+self.clip_epsilon) * advantage_attack, ratio*advantage_attack)
            L2 = torch.min(torch.clamp(ratio, 1.0-self.clip_epsilon, 1.0+self.clip_epsilon) * advantage_victim, ratio*advantage_victim)
            policy_loss = torch.mean((L1-L2)) #负号?

            value_loss = torch.mean(F.mse_loss(self.value(bn_s), td_target.detach()))
            value_ins_loss = torch.mean(F.mse_loss(self.value_ins(bn_s),reward_intrinsic.detach()+self.gamma*self.value_ins(bn_s_)*(1-bn_d)))

            self.optim.zero_grad()
            self.value_optim.zero_grad()
            self.value_ins_optim.zero_grad()
            policy_loss.backward()
            value_loss.backward()
            value_ins_loss.backward()
            self.optim.step()
            self.value_optim.step()
            self.value_ins_optim.step()

        # reset buffer
        replay_buffer.reset_buffer()
        # print(f'Policy Loss: {policy_loss.item()}, Value Loss: {value_loss.item()}, Value Ins Loss: {value_ins_loss.item()}')

    def save_model(self, episode):
        states = {
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'value_ins': self.value_ins.state_dict(),
            'surr_net': self.cd.surr_model.state_dict(),
            'RND_tar': self.cd.rnd.target_net.state_dict(),
            'RND_pre': self.cd.rnd.predictor_net.state_dict()
        }
        filepath = os.path.join(self.model_path, f'model.ppo_cd.{self.model_id}.{episode:04}.torch')
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
        filepath = os.path.join(self.model_path, f'model.ppo_cd.{self.model_id}.{episode:04}.torch')
        if os.path.isfile(filepath):
            self.logger.log(f'>> Loading scenario policy {self.name} model from {filepath}')
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)
            self.policy.load_state_dict(checkpoint['policy'])
            self.value.load_state_dict(checkpoint['value'])
            self.value_ins.load_state_dict(checkpoint['value_ins'])
            self.cd.surr_model.load_state_dict(checkpoint['surr_net'])
            self.cd.rnd.target_net.load_state_dict(checkpoint['RND_tar'])
            self.cd.rnd.predictor_net.load_state_dict(checkpoint['RND_pre'])
            self.continue_episode = episode
        else:
            self.logger.log(f'>> No scenario policy {self.name} model found at {filepath}', 'red')
            # exit()
