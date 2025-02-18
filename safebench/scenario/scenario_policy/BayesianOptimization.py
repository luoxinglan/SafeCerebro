''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-05 14:55:02
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''
from fnmatch import fnmatch
import numpy as np
import os
from safebench.scenario.scenario_policy.base_policy import BasePolicy
from scipy.stats import norm
from scipy.optimize import minimize

class BO(BasePolicy):
    name = 'BayesianOptimization'
    type = 'init_state'

    """ This agent is used for scenarios that do not have controllable agents. """
    def __init__(self, config, logger):
        self.logger = logger
        self.num_scenario = config['num_scenario']
        self.batch_size = config['batch_size']
        self.continue_episode = 0
        self.num_scenario_x = config['num_scenario_x']
        self.bounds = np.array([[-1, 1] for _ in range(self.num_scenario_x)])
        self.n_init = config['n_init']
        self.model_path = config['model_path']
        self.scenario_id = config['scenario_id']
        self.route_id = config['route_id']
        
        # 存储采样点和观测值
        self.X = []
        self.y = []
        
        # 高斯过程的超参数
        self.theta = 1.0
        self.sigma = 1.0
        self.n_iter = config['n_iter']
        self.i_iter = 0
        self.init_check = False
        
    def kernel(self, x1, x2):
        """RBF核函数"""
        return self.sigma * np.exp(-np.sum((x1-x2)**2) / (2*self.theta))
    
    def kernel_matrix(self, X1, X2):
        """计算核矩阵"""
        K = np.zeros((len(X1), len(X2)))
        for i, x1 in enumerate(X1):
            for j, x2 in enumerate(X2):
                K[i,j] = self.kernel(x1, x2)
        return K
    
    def gaussian_process(self, x):
        """高斯过程预测"""
        K = self.kernel_matrix(self.X, self.X)
        K_star = self.kernel_matrix(self.X, [x])
        K_star_star = self.kernel_matrix([x], [x])
        
        # 计算均值和方差
        mu = K_star.T @ np.linalg.inv(K) @ self.y
        sigma = K_star_star - K_star.T @ np.linalg.inv(K) @ K_star
        
        return mu[0], sigma[0,0]
    
    def expected_improvement(self, x):
        """计算期望改进"""
        mu, sigma = self.gaussian_process(x)
        sigma = max(sigma, 1e-9)  # 避免除零错误
        
        y_best = np.min(self.y)
        z = (y_best - mu) / sigma
        ei = sigma * (z * norm.cdf(z) + norm.pdf(z))
        
        return -ei  # 最小化负EI相当于最大化EI
        
    def train(self, replay_buffer):
        if self.init_check is not True:
            batch = replay_buffer.sample_init(self.batch_size)
            y = batch['episode_reward']
            self.y.append(y)
        if self.init_check is not True and len(self.y)>=self.n_init:
            self.init_check = True
            self.y = np.array(self.y)
        if self.init_check and self.i_iter <= self.n_iter: #初始化随机点完毕
            if len(self.X) > len(self.y):
                batch = replay_buffer.sample_init(self.batch_size)
                y_next = batch['episode_reward']
                self.y = np.append(self.y, y_next)
            self.x_next = None
            ei_max = float('inf')
            for _ in range(5):
                x0 = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
                res = minimize(self.expected_improvement, x0, 
                             bounds=self.bounds,
                             method='L-BFGS-B')
                
                if res.fun < ei_max:
                    self.x_next = res.x
                    ei_max = res.fun
            self.i_iter += 1
            
        if self.i_iter >= self.n_iter:
            # 返回最优解
            self.best_idx = np.argmin(self.y)
            self.logger.log(f"Optimal Solution: x = {self.X[self.best_idx]}, f(x) = {self.y[self.best_idx]}", color='blue')
            self.save_model(self.n_iter+self.n_init)
        
        replay_buffer.reset_init_buffer()
        return

    def set_mode(self, mode):
        self.mode = mode

    def get_action(self, state, infos, deterministic=False):
        return [None] * self.num_scenario

    def get_init_action(self, scenario_config, deterministic=False):
        if self.mode == 'eval':
            x = self.loaded_array
            return [x] * self.num_scenario, None
        
        if self.init_check is not True:
            x = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
            self.X.append(x)
        if self.init_check is not True and len(self.y)>=self.n_init:
            self.init_check = True
            self.X = np.array(self.X)
        if self.init_check and self.i_iter <= self.n_iter: #初始化随机点完毕
            x = self.x_next
            # 更新数据
            self.X = np.vstack((self.X, self.x_next))
        elif self.i_iter > self.n_iter:
            x = self.X[self.best_idx]
            
        return [x] * self.num_scenario, None

    def load_model(self, episode=None):
        if episode is None:
            episode = -1
            for _, _, files in os.walk(self.model_path):
                for name in files:
                    if fnmatch(name, "*npy"):
                        cur_episode = int(name.split(".")[-2])
                        if cur_episode > episode:
                            episode = cur_episode
        model_filename = os.path.join(self.model_path, f'model.BayesianOptimization.{self.scenario_id}.{self.route_id}.{episode:04}.npy')
        if os.path.isfile(model_filename):
            self.loaded_array = np.load(model_filename)
            self.continue_episode = episode
        else:
            self.logger.log(f'>> Fail to find bo model from {model_filename}', color='yellow')
        return self.continue_episode
        # return None

    def save_model(self, episode):
        if not os.path.exists(self.model_path):
            self.logger.log(f'>> Creating folder for saving model: {self.model_path}')
            os.makedirs(self.model_path)
        model_filename = os.path.join(self.model_path, f'model.BayesianOptimization.{self.scenario_id}.{self.route_id}.{episode:04}.npy')
        self.logger.log(f'>> Saving BayesianOptimization model to {model_filename}')
        np.save(model_filename, self.X[self.best_idx])
        