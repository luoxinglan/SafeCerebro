''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-22 17:26:29
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

# 由于局部优化性，很难收敛到碰撞的情况，极有可能是梯度太大了！

import os
import numpy as np
from fnmatch import fnmatch

from safebench.util.torch_util import CUDA, CPU

class GD():
    name = 'GradDescent'
    type = 'init_state'

    def __init__(self, scenario_config, logger):
        self.logger = logger
        self.continue_episode = 0
        self.num_scenario = scenario_config['num_scenario']
        self.lr = scenario_config['lr']

        self.f = np.zeros_like([0.0,0.0])
        self.history = []
        self.x = scenario_config['initial_point'] # [速度差异，位置差异]
        self.grad = np.zeros_like(self.x)
        self.epsilon = scenario_config['epsilon']
        self.grad_ind = 0
        self.MAX = False

    # 使用有限差分法来近似梯度
    def approximate_gradient(f, x, epsilon=1e-4):
        grad = np.zeros_like(x)
        for i in range(len(x)):
            # 创建一个扰动向量
            e = np.zeros_like(x)
            e[i] = epsilon
            # 计算f(x + epsilon * e_i) - f(x) / epsilon
            grad[i] = (f(x+e) - f(x-e)) / (2 * epsilon)
        return grad

    def get_init_action(self, state, deterministic=False):
        # 根据梯度更新参数
        e = np.zeros_like(self.x)
        e[self.grad_ind] = self.epsilon
        if self.MAX == False:
            self.x_tmp = self.x + e
            self.MAX = True
        else:
            self.x_tmp = self.x - e
            self.MAX = False
            self.grad_ind += 1
        
        action_list = self.x_tmp
        return [action_list], None
    
    def get_action(self, state, infos, deterministic=False):
        return [None] * self.num_scenario

    # train on batched data # 二维梯度下降
    def train(self, replay_buffer):
        if self.MAX == True:
            self.f[0] = replay_buffer.buffer_episode_reward[-1]/100
        else:
            self.f[1] = replay_buffer.buffer_episode_reward[-1]/100
            self.grad[self.grad_ind-1] = (self.f[0]-self.f[1])/(2*self.epsilon)

        if self.grad_ind == len(self.x):
            self.x -= self.lr * self.grad
            # 保存历史记录
            self.history.append(self.x.copy())
            self.grad_ind = 0
            self.grad = np.zeros_like(self.x)

    def save_model(self,epoch):
        pass

    def load_model(self, episode=None):
        pass
    
    def set_mode(self, mode):
        self.mode = mode
        pass