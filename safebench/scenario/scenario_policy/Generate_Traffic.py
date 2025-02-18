''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-05 14:55:02
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''
import subprocess

class generate_traffic:
    name = 'dummy'
    type = 'unlearnable'

    """ This agent is used for scenarios that do not have controllable agents. """
    def __init__(self, config, logger):
        self.logger = logger
        self.logger.log('>> This scenario does not require policy model, using a dummy one', color='yellow')
        self.num_scenario = config['num_scenario']
        self.continue_episode = 0

    def train(self, replay_buffer):
        pass

    def set_mode(self, mode):
        self.mode = mode

    def get_action(self, state, infos, deterministic=False):
        return [None] * self.num_scenario

    def get_init_action(self, scenario_config, deterministic=False):
        return [None] * self.num_scenario, None

    def load_model(self, episode=None):
        if episode:
            self.continue_episode = episode
        return self.continue_episode
        # return None

    def save_model(self, episode):
        return