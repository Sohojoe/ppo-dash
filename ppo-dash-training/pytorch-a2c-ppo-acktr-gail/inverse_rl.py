import os
import gym
import numpy as np
from collections import deque
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)
from baselines.common.atari_wrappers import LazyFrames
import time
from timeit import default_timer as timer
from sohojoe_monitor import detect_monitor_files
import json

class InverseRL(gym.Wrapper):
    def __init__(self, env, training_dir, test_mode=False):
        super(InverseRL, self).__init__(env)
        self._training_dir = training_dir
        self._test_mode = test_mode
        self._replay = None
        self._last_floor = -1
        self._last_seed = -1

    def step(self, action):
        if self.get_internal_floor != self._last_floor or self.get_seed != self._last_seed:
            self._init_floor()
        
        has_ended = False
        if self._end_step:
            has_ended = self._step_num >= self._end_step

        if self._replay and not has_ended:
            action = self._get_action()
        obs, rew, done, info = self.env.step(action)
        self._step_num += 1
        return obs, rew, done, info
    
    def _get_action(self):
        if self._cur_action[1] < 1:
            actions = self._replay.get('actions')
            self._cur_action = actions[self._action_idx]
            if self._action_idx < len(actions)-1:
                self._action_idx += 1
            
        action = self._cur_action[0]
        self._cur_action = (self._cur_action[0], self._cur_action[1]-1)
        return action

    def _init_floor(self):
        floor = self.get_internal_floor
        seed = self.get_seed
        self._last_floor = floor
        self._last_seed = seed
        self._replay = self._get_replay()
        self._step_num = 0
        self._action_idx = 0
        self._cur_action = (0,0)
        self._end_step = None
        if self._replay:
            inverse_rl = self._replay.get('inverse_rl', None)
            actions = self._replay.get('actions', None)
            a = np.asarray(actions)
            total_steps = a.sum(axis=0)[1]
            if inverse_rl:
                rnd_in = inverse_rl[0][0] * 10 # 10 fps
                rnd_out = inverse_rl[0][1] * 10 # 10 fps
                if True: # rand to end
                    rnd_out = total_steps
                rnd_size = rnd_out-rnd_in
                rnd = np.random.rand()
                from math import pow
                rnd = pow(rnd, 3)
                rnd = rnd * rnd_size
                rnd = rnd_out-rnd
                self._end_step = int(rnd)

        if self._replay:
            reset_floor = self.env.get_floor
            self.env.floor(self.get_internal_floor)
            self.env.unwrapped.reset()
            self.env.floor(reset_floor)
            no_action = action = self.env.action_space.sample()
            if len(self.env.action_space.shape) > 0:
                for i in range(self.env.action_space.shape[0]):
                    no_action[i] = 0
            else:
                no_action = 0
            for i in range(3):
                obs, rew, done, info = self.env.step(no_action)
                time.sleep(0.01) 


    def reset(self, **kwargs):
        self._last_floor = -1
        self._last_seed = -1
        observation = self.env.reset(**kwargs)
        return observation

    def _get_replay(self):
        floor = self.get_internal_floor
        seed = self.get_seed
        monitor_files = detect_monitor_files(self._training_dir)
        match_str = '.floor{:02}.seed{:02}.meta.json'.format(floor, seed)
        matches = [f for f in monitor_files if f.endswith(match_str)]
        while True:
            if len(matches) is 0:
                return None
            file_name = np.random.choice(matches)
            
            with open(file_name, 'r') as f:
                replay = json.load(f)
            actions = replay.get('actions', None)
            inverse_rl = replay.get('inverse_rl', None)
            if inverse_rl is None or len(inverse_rl) is 0:
                print('InverseRL error: missing inverse_rl', file_name)
                inverse_rl = None
            if actions is None or len(actions) is 0:
                print('InverseRL error: missing events', file_name)
                actions = None
            if self._test_mode and actions:
                return replay
            elif actions and inverse_rl:
                return replay
            matches.remove(file_name)

