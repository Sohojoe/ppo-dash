from sohojoe_obstacle_tower_env import ObstacleTowerEnv
from sohojoe_wrappers import KeyboardControlWrapper, RenderObservations, RetroWrapper
from otc_preprocessing import OTCPreprocessing
import numpy as np
import pandas as pd
import gym
from sohojoe_monitor import Monitor
import os, time
from inverse_rl import InverseRL

class DoubleResetWrapper(gym.Wrapper):
    def __init__(self, env):
        super(DoubleResetWrapper, self).__init__(env)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and rew >= 1.:
            print ("--- Success,", self.get_floor, self.get_seed)
        elif done:
            print ("--- Failed,", self.get_floor, self.get_seed)
        return obs, rew, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)

        no_action = action = self.env.action_space.sample()
        for i in range(self.env.action_space.shape[0]):
            no_action[i] = 0
        for i in range(3):
            obs, rew, done, info = self.env.step(no_action)
            time.sleep(0.01) 

        observation = self.env.reset(**kwargs)
        return observation
        

class LevelSelectorWrapper(gym.Wrapper):
    def __init__(self, env, reset_on_floor_complete=False, repeate_on_death=True, levels=None):
        super(LevelSelectorWrapper, self).__init__(env)
        self.levels = levels
        self._level = np.random.randint(0,25)
        self._seed = np.random.randint(0,100)
        self._reset_on_floor_complete = reset_on_floor_complete
        self._repeate_on_death = repeate_on_death

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if rew >= 1. and self._reset_on_floor_complete:
            done = True
            if self._repeate_on_death:
                task = self.levels[0]
                self.levels.remove(task)
        return obs, rew, done, info

    def reset(self, **kwargs):
        if self.levels is None:
            txt_input = input("Enter level ("+str(self._level)+"): ")
            self._level = int(txt_input) if len(txt_input) > 0 else self._level
            txt_input = input("Enter seed ("+str(self._seed)+"): ")
            self._seed = int(txt_input) if len(txt_input) > 0 else self._seed
        elif len(self.levels) is 0:
            self._level = np.random.randint(0,25)
            self._seed = np.random.randint(0,100)
        else:
            task = self.levels[0]
            self._level = task[0]
            self._seed = task[1]
            if not self._repeate_on_death:
                self.levels.remove(task)
        self.env.seed(self._seed)
        self.env.floor(self._level)
        observation = self.env.reset(**kwargs)
        return observation
class EventWrapper(gym.Wrapper):
    def __init__(self, env):
        super(EventWrapper, self).__init__(env)
        self._level = np.random.randint(0,25)
        self._seed = np.random.randint(0,100)
        txt_input = input("Enter level ("+str(self._level)+"): ")
        self._level = int(txt_input) if len(txt_input) > 0 else self._level
        txt_input = input("Enter seed ("+str(self._seed)+"): ")
        self._seed = int(txt_input) if len(txt_input) > 0 else self._seed

    def _load(self):
        # self._df = pd.read_csv('events.csv', encoding ='latin1')
        self._df = pd.read_csv('events.csv', encoding ='utf-8')

    def _save(self):
        self._df.to_csv('events.csv', encoding='utf-8', index=False)

    def _save_events(self):
        self._load()
        row = [self._level, self._seed]
        row.extend(self.events)
        while len(row) < len(self._df.columns):
            row.extend([np.NaN])
        row = pd.Series(row, index=self._df.columns)
        # row = pd.Series(row)
        self._df = self._df.append(row, ignore_index=True)
        self._df = self._df.drop_duplicates()
        print(self._df)
        print('-')
        self._save()

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        event = ''
        new_health = float(info['brain_info'].vector_observations[0][-1])
        # new_keys = int(info['brain_info'].vector_observations[0][:-1].sum())
        new_keys = int(info['brain_info'].vector_observations[0][1:-1].sum())
        picked_up_key = True if new_keys > self._last_keys else False
        used_key = True if new_keys < self._last_keys else False
        picked_up_health = True if new_health > self._last_health else False
        if self._episode_step is 0 and rew > 0:
            rew = 0
        self._episode_step += 1
        if rew > 0:
            if rew >= 1:
                done = True
                event = 'CompleteLevel'
            elif used_key:
                event = 'UnlockDoor'
            elif picked_up_key:
                event = 'FindKey'
            elif self._last_event is 'UnlockDoor' or self._last_event is 'OpenDoor' or self._last_event is 'PushBlockOnToPad':
                event = 'InnerDoor'
            else:
                while len(event) is 0:
                    int_input = input("Enter event \n" \
                        " [0] = OpenDoor\n" \
                        " [1] = PushBlockOnToPad\n :" \
                        " [2] = InnerDoor\n :")
                    try:
                        int_input = int(int_input)
                    except ValueError:
                        int_input = -1
                    if int_input is 0:
                        event = 'OpenDoor'
                    elif int_input is 1:
                        event = 'PushBlockOnToPad'
                    elif int_input is 2:
                        event = 'InnerDoor'
        if len(event) > 0:
            self.events.append(event)
            self._last_event = event
        self._last_health = new_health
        self._last_keys = new_keys
        if done:
            print ('level: ', self._level, 'seed: ', self._seed, self.events)
            self._save_events()
            self._seed = self._seed + 1 if self._seed < 99 else 0
            if self._seed is 0:
                self._level = self._level + 1 if self._level < 24 else 0
            # self._level = self._level + 1 if self._level < 24 else 11
            # if self._level is 0:
            #     self._seed = self._seed + 1 if self._seed < 99 else 0
            # self._seed += 1
            # if self._seed > 99:
            #     self._seed = 0
            #     self._level += 1
            #     if self._level > 24:
            #         self._level = 0
        return obs, rew, done, info

    def reset(self, **kwargs):
        # txt_input = input("Enter level ("+str(self._level)+"): ")
        # self._level = int(txt_input) if len(txt_input) > 0 else self._level
        # txt_input = input("Enter seed ("+str(self._seed)+"): ")
        # self._seed = int(txt_input) if len(txt_input) > 0 else self._seed
        self.env.seed(self._seed)
        self.env.floor(self._level)
        self.events = []
        self._last_health = new_health = 9999999.
        self._last_keys = new_keys = 0
        self._last_event = ''
        self._episode_step = 0
        observation = self.env.reset(**kwargs)
        return observation

import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

level24_list = [
    (24, 0), (24, 1), (24, 2), (24, 3), (24, 4), 
    (24, 5), (24, 6), (24, 7), (24, 8), (24, 9), 
    (24, 10), (24, 11), (24, 12), (24, 13), (24, 14), 
    (24, 15), (24, 16), (24, 17), (24, 18), (24, 19), 
    (24, 20), (24, 21), (24, 22), (24, 23), (24, 24), 
    (24, 25), (24, 26), (24, 27), (24, 28), (24, 29), 
]
quick_list = [
    (5,	11),(5,	25),(5,	15),(6,	72),(5,	1),
    (5,	40),(5,	81),(6,	20),(6,	91),(6,	14),
    (10, 43), (10, 53), (10, 59), (10, 61), (10, 63), 
    (10, 64), (10, 65), (10, 67), (10, 68), (10, 69), 
]

struggling_list = [
    (5,	11),(5,	25),(5,	15),(6,	72),(5,	1),
    (5,	40),(5,	81),(6,	20),(6,	91),(6,	14),
    (8,	0),(7,	31),(6,	37),(5,	89),(7,	92),
    (7,	47),(5,	44),(9,	61),(7,	30),(9,	22),
    (7,	11),(9,	25),(7,	15),(8,	1), (7,	40),
    (9,	81),(8,	20),(7,	91),(8,	31),(6,	89),
    (9,	92),(6,	61),(8,	30),(6,	22),(6,	11),
    (7,	25),(9,	15),(7,	1), (6,	81),(7,	20),
    (5,	31),(9,	89),(7,	61),(9,	30),(7,	22),
    (9,	11),(9,	1), (8,	22),
]

full_list = [
    (5,	11),(5,	25),(5,	15),(6,	72),(5,	1),
    (5,	40),(5,	81),(6,	20),(6,	91),(6,	14),
    (10, 43), (10, 53), (10, 59), (10, 61), (10, 63), 
    (8,	0),(7,	31),(6,	37),(5,	89),(7,	92),
    (7,	47),(5,	44),(9,	61),(7,	30),(9,	22),
    (10, 64), (10, 65), (10, 67), (10, 68), (10, 69), 
    (7,	11),(9,	25),(7,	15),(8,	1), (7,	40),
    (9,	81),(8,	20),(7,	91),(8,	31),(6,	89),
    (10, 71), (10, 74), (10, 80), (10, 82), (10, 83), 
    (9,	92),(6,	61),(8,	30),(6,	22),(6,	11),
    (7,	25),(9,	15),(7,	1), (6,	81),(7,	20),
    (10, 84), (10, 86), (10, 87), (10, 89), (10, 90), 
    (5,	31),(9,	89),(7,	61),(9,	30),(7,	22),
    (9,	11),(9,	1), (8,	22),
    (10, 91), (10, 93), (10, 96), (10, 98), (10, 99),
]


redos = [
    (0, 3),
    (10, 84), (10, 86), (10, 87), (10, 89), (10, 90), 
    (10, 91), (10, 93), (10, 96), (10, 98), (10, 99)
]

alt_level10_list = [
    # (0, 3),
    (10, 11), (10, 25),(10, 15), (10, 72) ,(10, 1),
    (10, 40),(10, 81),(10, 20),(10, 14), (10, 0),
    (10, 31),(10, 37),(10, 92),(10, 47),(10, 44),
    (10, 30),(10, 22),
]

level10_list = [
    (0, 3),
    (10, 36), (10, 41),  (10, 42), 
    (10, 43), (10, 53), (10, 59), (10, 61), (10, 63), 
    (10, 64), (10, 65), (10, 67), (10, 68), (10, 69), 
    (10, 71), (10, 74), (10, 80), (10, 82), (10, 83), 
    (10, 84), (10, 86), (10, 87), (10, 89), (10, 90), 
    (10, 91), (10, 93), (10, 96), (10, 98), (10, 99)
]
seed_0 = [
    (0,0), (1,0), (2,0), (3,0), (4,0), 
    (5,0), (6,0), (7,0), (8,0), (9,0), 
    (10,0), (11,0), (12,0), (13,0), (14,0), 
    (15,0), (16,0), (17,0), (18,0), (19,0), 
    (20,0), (21,0), (22,0), (23,0), (24,0), 
]
all_seed_list = [
    (0, 11), (0, 25), (0, 15), (0, 72), (0, 1), 
    (0, 40), (0, 91), (0, 20), (0, 81), (0, 14), 
    (0, 31), (0, 37), (0, 0), (0, 92), (0, 61), 
    (0, 47), (0, 74), (0, 44), (0, 48), (0, 22), 
    (0, 67), (0, 89), (0, 30), (0, 58), (0, 42), 
    (0, 23), (0, 28), (0, 7), (0, 68), (0, 86), 
    (0, 33), (0, 36), (0, 78), (0, 79), (0, 51), 
    (0, 64), (0, 62), (0, 77), (0, 38), (0, 96), 
    (0, 88), (0, 70), (0, 6), (0, 98), (0, 93), 
    (0, 18), (0, 97), (0, 35), (0, 95), (0, 13), 
    (0, 39), (0, 16), (0, 57), (0, 87), (0, 90), 
    (0, 34), (0, 26), (0, 9), (0, 19), (0, 27), 
    (0, 82), (0, 73), (0, 63), (0, 99), (0, 2), 
    (0, 17), (0, 41), (0, 71), (0, 65), (0, 43), 
    (0, 12), (0, 49), (0, 66), (0, 3), (0, 60), 
    (0, 46), (0, 8), (0, 59), (0, 76), (0, 55), 
    (0, 84), (0, 83), (0, 21), (0, 56), (0, 69), 
    (0, 29), (0, 4), (0, 53), (0, 80), (0, 85), 
    (0, 10), (0, 32), (0, 5), (0, 45), (0, 75), 
    (0, 54), (0, 24), (0, 50), (0, 94), (0, 52)   
]

record = False
# record = True

# os.environ['OTC_EVALUATION_ENABLED'] = 'True'
is_grading = os.getenv('OTC_EVALUATION_ENABLED', False)
# is_grading = False
# env = ObstacleTowerEnv('./ObstacleTower/obstacletower', retro=True, worker_id=1001, realtime_mode=True)
env = ObstacleTowerEnv(
    '../ObstacleTower/obstacletower', 
    # retro=True, 
    retro=False, 
    # worker_id=1001,
    worker_id=0,
    timeout_wait=6000,
    # realtime_mode=True
    )
env = DoubleResetWrapper(env)
# env = EventWrapper(env)
if not is_grading:
    # env = LevelSelectorWrapper(env, reset_on_floor_complete=True, levels=level_list)
    # env = LevelSelectorWrapper(env, levels=seed_0, reset_on_floor_complete=True, repeate_on_death=True)
    env = LevelSelectorWrapper(env, levels=redos, reset_on_floor_complete=True, repeate_on_death=False)
    # env = LevelSelectorWrapper(env, levels=full_list, reset_on_floor_complete=True)
    # env = LevelSelectorWrapper(env, reset_on_floor_complete=True)
env = RetroWrapper(env, False, size=84*2)
env = OTCPreprocessing(env)
env = RenderObservations(env)

if record:
    if not is_grading:
        monitor = env = Monitor(env, './new_video')
else:
    env = InverseRL(env, './new_video')
env = KeyboardControlWrapper(env, set_human_control=True, no_diagnal=True)

while True:
    # level = 18
    # seed = 36
    env.reset()
    done = False

    while not done:
        while env.is_paused:
            env.render()
            time.sleep(0.1)
        env.render()
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
