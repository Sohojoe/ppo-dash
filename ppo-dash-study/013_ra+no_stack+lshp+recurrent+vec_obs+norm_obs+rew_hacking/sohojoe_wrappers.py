import os
import gym
import numpy as np
import pandas as pd
from collections import deque
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)
from baselines.common.atari_wrappers import LazyFrames
import time
from timeit import default_timer as timer


def done_grading(env):
    if hasattr(env, 'done_grading'):
        return env.done_grading()
    if hasattr(env, 'env'):
        return done_grading(env.env)
    if hasattr(env, '_env'):
        return done_grading(env._env)

def is_grading(env):
    if hasattr(env, 'is_grading'):
        return env.is_grading()
    if hasattr(env, 'env'):
        return is_grading(env.env)
    if hasattr(env, '_env'):
        return is_grading(env._env)


class ScoreLevelWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        # np.random.uniform generates uniformly
        seeds = [x for x in range(100)]
        # np.random.shuffle(seeds)
        self._scoreSeeds = seeds[95:100]
        self._scoreSeeds = [
            1, 25, 52, 61, 59,
            # 11, 25, 15,	72,	1,
            # 40, 81,	20, 91,	14,
            # 52, 43, 53, 59, 61, 
            # 63
            ]
        # self._levels = [0,1,2,3,4,5,6]
        self._runAchievedLevel = np.zeros((5,5))
        self._runAchievedScore = np.zeros((5,5))
        self._seedIdx = 0
        self._runIdx = 0
        self._level = 0
        self._max_level = 0
        self._ep_rew = 0
        # print ('score. Seeds', [n for n in self._scoreSeeds])
        self._setLevel()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self._ep_rew += reward
        message = ''

        if reward >= 1:
            message += 'completed level ' + str(self._level) + ', '
            self._level += 1
        elif reward > 0:
            message += 'got reward, '


        if done:
            # out of time
            message += 'game over, '
            self._runAchievedLevel[self._seedIdx,self._runIdx] = self._level
            self._runAchievedScore[self._seedIdx,self._runIdx] = self._ep_rew
            self._level = 0
            self._ep_rew = 0
            self._seedIdx += 1
            if self._seedIdx >= len(self._scoreSeeds):
                self._seedIdx = 0
                self._runIdx += 1
                if self._runIdx >= 5:
                    self._calc_results()

        if len(message) > 0:
            print (message + 'reward = ' + str(np.around(reward,1)))      

        # +0.1 is provided for opening doors, solving puzzles, or picking up keys.
        return ob, reward, done, info   

    def reset(self):
        self._setLevel()
        return self.env.reset()
    
    def _setLevel(self):
        self.env.floor(self._level)
        self.env.dense_reward = 1
        self._last_seed = self._scoreSeeds[self._seedIdx]
        # print ('seed:',self._last_seed)
        self.env.seed(self._last_seed)
    
    def _calc_results(self):
        for i in range(len(self._scoreSeeds)):
            scores = np.around(self._runAchievedScore[i,:], 2)
            av_score = np.around(np.average(self._runAchievedLevel), 2)
            print (
                'seed:', self._scoreSeeds[i], 
                'ave level:', np.average(self._runAchievedLevel[i,:]), self._runAchievedLevel[i,:], 
                'ave score:', av_score, scores)
        tot_ave_score = np.around(np.average(self._runAchievedScore), 2)
        print ('total ave level:', np.average(self._runAchievedLevel), 'total ave score:', tot_ave_score)


class TrainLevelWrapper(gym.Wrapper):
    """
    Send the root node's parameters to every worker.

    Arguments:
      env: the gym env to wrap
      reset_mode: 
            '0'=restart at 0, 
            'rand'=pick a random level
            'seed_from_10'=start level 10 with one seed then do mode
            'alfie'= focused training mode
    """
    def __init__(self, env, agent_id, reset_mode='0', show_debug=False, reward_hacking=False):
        gym.Wrapper.__init__(self, env)
        self._show_debug = show_debug
        self._reset_mode=reset_mode
        self._agent_id=agent_id
        self._stage_reward_mapping = True
        self._reward_hacking = reward_hacking
        # self._done_on_block_stage = self._reset_mode is 'seed_from_10'
        self._done_on_block_stage = False
        # np.random.uniform generates uniformly
        seeds = [x for x in range(100)]
        # np.random.shuffle(seeds)
        # self._trainSeeds = seeds[0:95]
        self._alfie_list = [
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
        self._easy10_seeds = [
            # 2, 3, 4, 12, 17, 
            # 24, 29, 42, 43, 53, 
            # 61, 65, 69, 80, 82, 
            # 84, 89, 90, 91
            1,  2, 3, 4, 12, 
            16, 17, 21, 24, 26,
            29,34,36,41,42,
            43,53,59,61,63,
            64,65,67,68,69,
            71,74,80,82,83,
            84,86,87,89,90,
            91,93,96,98,99
        ]
        # self._trainSeeds = [
        #     11, 25, 15,	72,	1,
        #     40, 81,	20, 91,	14,

        #     52, 52,
        #     43, 53, 59, 61, 63,
        # ]
        self._trainSeeds = [
            0, 1, 2, 3, 4, 
            5, 6, 7, 8, 9, 
            10, 11, 12, 13, 14, 
            15, 16, 17, 18, 19, 
            20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 
            30, 31, 32, 33, 34, 
            35, 36, 37, 38, 39, 
            40, 41, 42, 43, 44, 
            45, 46, 47, 48, 49, 
            50, 51, 52, 53, 54, 
            55, 56, 57, 58, 59, 
            60, 61, 62, 63, 64, 
            65, 66, 67, 68, 69,
            70, 71, 72, 73, 74, 
            75, 76, 77, 78, 79, 
            80, 81, 82, 83, 84, 
            85, 86, 87, 88, 89, 
            90, 91, 92, 93, 94,
            95, 96, 97, 98, 99,
            ]
        self._level = 0
        if reset_mode is 'seed_from_10':
            # self._all_trainSeeds = self._trainSeeds
            # self._trainSeeds = self._all_trainSeeds[:1]
            # self._repeat_every_seed=10
            # self._repeats_to_go=self._repeat_every_seed
            # self._level=10
            self._level=10
            # self._trainSeeds = self._easy10_seeds
        if reset_mode is 'alfie':
            i = np.random.randint(0,len(self._alfie_list))
            t = self._alfie_list[i]
            self._trainSeeds = [t[1]]
            self._level=t[0]
            print(t)
        self._max_level = 0
        # print ('train. Seeds', [n for n in self._trainSeeds])
        self._setLevel()
        self._true_reward = 0
        self._true_steps = 0
        self._start_time = timer()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)

        if self._true_steps is 0 and reward > 0:
            reward = 0

        self._true_reward += reward
        self._true_steps += 1

        new_health = float(ob[2])
        new_keys = int(ob[1].sum())
        internal_reward = reward
        reward_hack = reward

        info_str = ''
        show_debug = self._show_debug

        if done:
            info_str += 'gameover on level:' + str(self._level) + ', '
            info_str += 'seed:' + str(self.get_seed) + ', '
            self._last_health = 99999. # force high
            if 'epsiode' not in info:
                 info['episode']={}
            info['episode']['r'] = self._true_reward
            info['episode']['l'] = self._true_steps
            info['episode']['t'] = timer() - self._start_time
            info['episode']['seed'] = self.get_seed
            info['episode']['floor'] = self._level
            info['episode']['start'] = self.get_floor
            info['episode']['agent'] = self._agent_id
            # out of time
            reward_hack = -1
            self._level = 0
            rnd = np.random.uniform(0.,1.) ** 2
            # self._level = int(self._max_level * rnd) # focus on early levels
            # self._level = int(self._max_level * np.random.uniform(0.,1.)) # focus equally across all levels
            # self._level = self._max_level - int(self._max_level * rnd) # focus on later levels
            if self._reset_mode is 'rand':
                self._level = np.random.randint(0,25)
            elif self._reset_mode is 'seed_from_10':
                self._level = 10 # focus on level 10
            if self._reset_mode is 'alfie':
                i = np.random.randint(0,len(self._alfie_list))
                t = self._alfie_list[i]
                self._trainSeeds = [t[1]]
                self._level=t[0]
                print(t)
            self._true_reward = 0
            self._true_steps = 0
            show_debug = True
            
        # +0.1 is provided for opening doors, solving puzzles, or picking up keys.
        elif reward >= 1:
            info_str += 'completed level: ' + str(self._level) + ', '
            # done = True
            # bonus for remaining time
            reward_hack += (new_health / 10000)
            # if self._reset_mode is 'seed_from_10':
            #     self._level = 10
            #     # idx=len(self._trainSeeds)
            #     # if self._last_seed is self._trainSeeds[idx-1]:
            #     #     self._repeats_to_go-=1
            #     #     if self._repeats_to_go is 0:
            #     #         self._repeats_to_go=self._repeat_every_seed
            #     #         idx=idx+1
            #     #         if idx > len(self._all_trainSeeds):
            #     #             idx=len(self._all_trainSeeds)
            #     #         self._level += 1
            #     #     self._trainSeeds=self._all_trainSeeds[:idx]
            # else:
            #     self._level += 1
            self._level += 1
            if self._level > self._max_level:
                self._max_level = self._level
                print ('new max level:', self._max_level)
        elif reward > 0:
            if self._block_stage == self._stage:
                reward_hack += 1.
                reward_hack += (new_health / 10000)
                info_str += '---- Solve Push Block Puzzel ----, ' 
                info_str += 'Level:' + str(self.get_floor) + ', '
                info_str += 'Seed:' + str(self.get_seed) + ', '
                info_str += 'internal_reward:' + str(np.around(internal_reward,1))+ ', '
                show_debug = True
                if self._done_on_block_stage:
                    done = True
            else:
                info_str += 'got reward, ' 
            self._stage += 1
        elif len(ob) == 3 and new_health > self._last_health:
            info_str += 'found health, ' 
            reward_hack = .1 # for pickup
        if new_keys > self._last_keys:
            info_str += 'found key, ' 
            reward_hack = 1
        self._last_health = new_health
        self._last_keys = new_keys

        if show_debug and len(info_str) > 0:
            if done:
                print ('  -----', info_str, '-----')
            else:
                print (info_str + 'reward_hack: ' + str(np.around(reward_hack,1)))
        # TODO reward_hack
        if self._reward_hacking:
            reward = reward_hack
        return ob, reward, done, info   

    def reset(self):
        self._setLevel()
        obs = self.env.reset()
        self._block_stage = None
        if self._stage_reward_mapping:
            self._stage = 1
            df = pd.read_csv('events.csv', dtype={'Level': np.int,'Seed':np.int})
            df = df[df['Step03'] == 'PushBlockOnToPad']
            df = df[df['Level'] == self.env.get_floor]
            df = df[df['Seed'] == self.env.get_seed]
            if df.shape[0] > 0:
                self._block_stage = 3
        return obs
    
    def _setLevel(self):
        # self._level = 10
        # self._trainSeeds = [1,20,22,30,72,81]
        self._last_health = 99999. # force high
        self._last_keys = 0
        self.env.floor(self._level)
        self.env.dense_reward = 1
        ok = False
        self._last_seed = np.random.choice(self._trainSeeds)
        # while not ok:
        #     self._last_seed = np.random.choice(self._trainSeeds)
        #     bug1 = self._last_seed == 58 and self._level == 10
        #     bug2 = self._last_seed == 72 and self._level == 12
        #     bug3 = self._last_seed == 37 and self._level == 11
        #     bug4 = self._last_seed == 72 and self._level == 9
        #     if not bug1 and not bug2 and not bug3 and not bug4: ok = True 
        self.env.seed(self._last_seed)

    @property
    def get_internal_floor(self):
        return self._level

class RenderObservations(gym.Wrapper):
    def __init__(self, env, display_vector_obs=True):
        gym.Wrapper.__init__(self, env)
        self.viewer = None
        self._empty = np.zeros((1,1,1))
        self._has_vector_obs = hasattr(self.observation_space, 'spaces')
        self._8bit = None
        self._display_vector_obs = display_vector_obs
        
    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        should_render = True
        if 'human_agent_display' in globals():
            global human_agent_display
            should_render = human_agent_display
        self._renderObs(ob, should_render)
        return ob, reward, done, info   

    def _renderObs(self, obs, should_render):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()
        if not should_render:
            self.viewer.imshow(self._empty)
            return self.viewer.isopen
        if self._has_vector_obs:
            visual_obs = obs['visual'].copy()
            vector_obs = obs['vector'].copy()
        else:
            visual_obs = obs.copy()           
        if self._has_vector_obs and self._display_vector_obs:
            w = 84
            # Displays time left and number of keys on visual observation
            key = vector_obs[0:-1]
            time_num = vector_obs[-1]
            key_num = np.argmax(key, axis=0)
            # max_bright = 1
            max_bright = 255
            visual_obs[0:10, :, :] = 0
            for i in range(key_num):
                start = int(i * 16.8) + 4
                end = start + 10
                visual_obs[1:5, start:end, 0:2] = max_bright
            visual_obs[6:10, 0:int(time_num * w), 1] = max_bright    
        self._8bit = visual_obs
        # if type(visual_obs[0][0][0]) is np.float32 or type(visual_obs[0][0][0]) is np.float64:
            # _8bit = (255.0 * visual_obs).astype(np.uint8)
        self._8bit = ( visual_obs).astype(np.uint8)
        self.viewer.imshow(self._8bit)
        return self.viewer.isopen

    def render(self, mode='human', **kwargs):
        if self.viewer:
            self.viewer.imshow(self._8bit)
        return self._8bit

    def reset(self):
        return self.env.reset()

    def close(self):
        self.env.close()
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

class FrameStackMono(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=3+(k-1))
        self.color_frames = deque([], maxlen=3)
        self.mono_frames = deque([], maxlen=k)
        self._has_vector_obs = hasattr(self.observation_space, 'spaces')
        if self._has_vector_obs:
            shp = env.observation_space.spaces['visual'].shape
            self.observation_space.spaces['visual'] = spaces.Box(
                low=env.observation_space.spaces['visual'].low.mean(), 
                high=env.observation_space.spaces['visual'].high.mean(), 
                shape=(shp[:-1] + (shp[-1] + k-1,)), 
                dtype=env.observation_space.spaces['visual'].dtype)
        else:
            shp = env.observation_space.shape
            self.observation_space = spaces.Box(
                low=env.observation_space.low.mean(), 
                high=env.observation_space.high.mean(), 
                shape=(shp[:-1] + (shp[-1] + k-1,)), 
                dtype=env.observation_space.dtype)

    def reset(self):
        if self._has_vector_obs:
            ob = self.env.reset()
            for _ in range(self.k):
                self._add_ob(ob['visual'])
            ob['visual'] = self._get_ob()
        else:
            ob = self.env.reset()
            for _ in range(self.k):
                self._add_ob(ob)
            ob = self._get_ob()
        return ob

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        if self._has_vector_obs:
            self._add_ob(ob['visual'])
            ob['visual'] = self._get_ob()
            return ob, reward, done, info
        else:
            self._add_ob(ob)
            return self._get_ob(), reward, done, info

    def _add_ob(self, ob):
        ob_t = ob.T
        self.color_frames.append(ob_t[0])
        self.color_frames.append(ob_t[1])
        self.color_frames.append(ob_t[2])
        mono = cv2.cvtColor(ob.astype(np.float32), cv2.COLOR_RGB2GRAY)
        mono = mono.astype(ob.dtype)
        self.mono_frames.append(mono)


    def _get_ob(self):
        # col = LazyFrames(list(self.color_frames))
        # mono = LazyFrames(list(self.mono_frames))
        self.frames.append(self.color_frames[0])
        self.frames.append(self.color_frames[1])
        self.frames.append(self.color_frames[2])
        for i in range(self.k-1):
            self.frames.append(self.mono_frames[i+1])
        # assert len(self.frames) == self.observation_space.shape[2]
        # output = LazyFrames(list(self.frames))
        output = np.array(list(self.frames)).T
        return output


class FrameStackMonoTemporal(gym.Wrapper):
    def __init__(self, env, k, temporal_frames):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.temporal_frames = temporal_frames
        self.frames = deque([], maxlen=3+len(self.temporal_frames))
        self.color_frames = deque([], maxlen=3)
        self.mono_frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=env.observation_space.low.mean(), 
            high=env.observation_space.high.mean(), 
            shape=(shp[:-1] + (shp[-1] + len(self.temporal_frames),)), 
            dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self._add_ob(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self._add_ob(ob)
        return self._get_ob(), reward, done, info

    def _add_ob(self, ob):
        ob_t = ob.T
        self.color_frames.append(ob_t[0])
        self.color_frames.append(ob_t[1])
        self.color_frames.append(ob_t[2])
        mono = cv2.cvtColor(ob, cv2.COLOR_RGB2GRAY)
        self.mono_frames.append(mono)


    def _get_ob(self):
        # col = LazyFrames(list(self.color_frames))
        # mono = LazyFrames(list(self.mono_frames))
        self.frames.append(self.color_frames[0])
        self.frames.append(self.color_frames[1])
        self.frames.append(self.color_frames[2])

        for i in self.temporal_frames:
            self.frames.append(self.mono_frames[i])
        assert len(self.frames) == len(self.temporal_frames)+3
        # output = LazyFrames(list(self.frames))
        output = np.array(list(self.frames)).T
        return output


num_actions = 0
human_wants_restart = False  
human_agent_action = 0   
human_sets_pause = False
human_has_control = False
human_agent_display = True
up_key = False
pause = False
class KeyboardControlWrapper(gym.ActionWrapper):
    def __init__(self, env, set_human_control=False, no_diagnal=False):
        super(KeyboardControlWrapper, self).__init__(env)   
        self._env = env
        self._set_human_control = set_human_control
        self._no_diagnal = no_diagnal
        self._last_action = 0

        # env.render()
        # env.unwrapped.viewer.window.on_key_press = key_press
        # env.unwrapped.viewer.window.on_key_release = key_release   
        global human_agent_action, human_wants_restart, human_sets_pause, human_has_control, num_actions, human_agent_display, up_key, pause
        num_actions = env.action_space.n
        human_wants_restart = False  
        human_agent_action = 0   
        human_sets_pause = False
        human_has_control = self._set_human_control
        human_agent_display = True
        up_key = False
        pause = False
        self._please_lazy_init = True
        self._last_time = time.time()

    def action(self, action):
        global human_agent_action, human_wants_restart, human_sets_pause, human_has_control, num_actions, human_agent_display, up_key, pause
        def key_press(key, mod):
            global human_agent_action, human_wants_restart, human_sets_pause, human_has_control, num_actions, human_agent_display, up_key, pause
            old_action=human_agent_action
            if key==0xff0d: human_wants_restart = True
            if key==65307: human_sets_pause = not human_sets_pause
            if key==65307: human_has_control = not human_has_control
            if key==65362: # up
                if human_agent_action == 3: human_agent_action = 6
                elif human_agent_action == 4: human_agent_action = 7
                else: human_agent_action = 1 
                up_key = True
            if key==65364: human_agent_action = 2 # down
            if key==65361: # left
                if human_agent_action == 1: human_agent_action = 6
                else: human_agent_action = 3
            if key==65363: # right
                if human_agent_action == 1: human_agent_action = 7
                else: human_agent_action = 4
            # if key==113: human_agent_action = 6 # q forward + left
            # if key==119: human_agent_action = 1 # w forward
            # if key==101: human_agent_action = 7 # e forward + right
            # if key==115: human_agent_action = 2 # s back
            # if key==97: human_agent_action = 3 # a back
            # if key==100: human_agent_action = 4 # d back
            if key==112: 
                pause = not pause # p pause
                print ('pause = ', pause)
                return
            if key==32: # space
                human_agent_action = 5 
            if key==65289: human_agent_display = not human_agent_display
            # 65307 # escape
            a = int( key - ord('0') ) 
            if a <= 0 or a >= num_actions: 
                if old_action is human_agent_action:
                    print ('key:', key)
                return
            human_agent_action = a

        def key_release(key, mod):
            global human_agent_action, num_actions, up_key
            # if key==65362: human_agent_action = 0 # up
            if key==65364 and human_agent_action==2: # down
                human_agent_action = 0 
            # if key==65361: human_agent_action = 0 # left
            # if key==65363: human_agent_action = 0 # right
            if key==65362: # up
                if human_agent_action == 6: human_agent_action = 3
                elif human_agent_action == 7: human_agent_action = 4
                elif human_agent_action == 1: human_agent_action=0
                up_key=False
            if key==65361: # left
                if human_agent_action == 6: human_agent_action = 1
                elif human_agent_action == 3: human_agent_action=0
            if key==65363: # right
                if human_agent_action == 7: human_agent_action = 1
                elif human_agent_action == 4: human_agent_action=0

            if key==32: # space
                if up_key:
                    human_agent_action = 1
                else:
                    human_agent_action = 0
            # if key==113: human_agent_action = 0 # q forward + left
            # if key==119: human_agent_action = 0 # w forward
            # if key==101: human_agent_action = 0 # e forward + right
            # if key==115: human_agent_action = 0 # s back
            # if key==97: human_agent_action = 0 # a back
            # if key==100: human_agent_action = 0 # d back
            
            a = int( key - ord('0') )
            if a <= 0 or a >= num_actions: return
            if human_agent_action == a:
                human_agent_action = 0
        if self._please_lazy_init and self._env.viewer is not None:
            self._please_lazy_init = False
            self._env.viewer.window.on_key_press = key_press
            self._env.viewer.window.on_key_release = key_release   
            human_has_control = self._set_human_control
        if human_has_control:
            # while pause:
            #     time.sleep(0.01)
            while time.time()-self._last_time < 1/10.:
                time.sleep(0.01)
            self._last_time = time.time()
            if self._no_diagnal:
                if human_agent_action == 6 or human_agent_action == 7:
                    human_agent_action = self._last_action
            a = human_agent_action
            self._last_action = human_agent_action
            # if human_agent_action is 3 or human_agent_action is 4:
            #     human_agent_action = 0
            return a
        return action

    @property 
    def is_paused(self):
        global pause
        return pause
    def render(self, mode='human'):
        return self._env.render(mode=mode)

class RetroWrapper(gym.ObservationWrapper):
    def __init__(self, env, randomize, size=84, keep_obs=False):
        super(RetroWrapper, self).__init__(env)
        self._randomize = randomize
        self._size = size
        self._is_otc = hasattr(self.observation_space, 'spaces')            
        depth = 3
        self._8bit = True
        # self._8bit = False
        self._keep_obs = keep_obs
        # if not self._keep_obs:
        image_space_max = 1.0
        image_space_dtype = np.float32
        if self._8bit:
            image_space_max = 255
            # image_space_dtype = np.uint8
        camera_height = size
        camera_width = size

        image_space = spaces.Box(
            0, image_space_max,
            dtype=image_space_dtype,
            shape=(camera_height, camera_width, depth)
        )
        if self._is_otc:
            self._spaces = (image_space,self.observation_space[1],self.observation_space[2])  
            self._vector_obs_size = self.observation_space[1].n + self.observation_space[2].shape[0]
            self._vector_time_idx = self.observation_space[1].n
            vector_obs_shape = spaces.Box(0, 1., dtype=np.float32, shape=([self._vector_obs_size]))
            self.observation_space = spaces.Dict({'visual':image_space, 'vector':vector_obs_shape})
        else:
            self.observation_space = image_space

    def _get_vector_obs(self):
        if self._is_otc:
            v = np.zeros(self._vector_obs_size, dtype=np.float32)
            v[self._key] = 1
            v[self._vector_time_idx] = self._time
            return v
        return None

    def observation(self, obs):
        w = self._size
        h = self._size
        # extract observations
        hd_visual_obs = obs[0]
        key = obs[1]
        time = obs[2]
        if self._randomize:
            if np.random.choice([0,1]):
                key = 5
                time = 10000
            else:
                key = 0
                time = 0
        self._key = key
        self._time = min(time, 10000) / 10000

        if hd_visual_obs.shape == self.observation_space.shape:
            visual_obs = hd_visual_obs
        else:
            # resize 
            # from PIL import Image
            # # hd_visual_obs = (255.0 * hd_visual_obs).astype(np.uint8)
            # obs_image = Image.fromarray(hd_visual_obs)
            # obs_image = obs_image.resize((84, 84), Image.NEAREST)
            # visual_obs = np.array(obs_image)
            # # visual_obs = (visual_obs).astype(np.float32) / 255.
            # obs_image = cv2.resize(hd_visual_obs, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
            # obs_image = cv2.resize(hd_visual_obs, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
            obs_image = cv2.resize(hd_visual_obs, dsize=(w, h), interpolation=cv2.INTER_AREA)
            # obs_image = cv2.resize(hd_visual_obs, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
            # obs_image = cv2.resize(hd_visual_obs, dsize=(w, h), interpolation=cv2.INTER_LANCZOS4)
            visual_obs = np.array(obs_image)

        if not self._keep_obs:
            # Displays time left and number of keys on visual observation
            # key = vector_obs[0:6]
            # time = vector_obs[6]
            # key_num = np.argmax(key, axis=0)
            key_num = self._key
            time_num = self._time

            max_bright = 1

            visual_obs[0:10, :, :] = 0
            for i in range(key_num):
                start = int(i * 16.8) + 4
                end = start + 10
                visual_obs[1:5, start:end, 0:2] = max_bright
            visual_obs[6:10, 0:int(time_num * w), 1] = max_bright

        if self._8bit:
            # visual_obs = (255.0 * visual_obs).astype(np.uint8)
            visual_obs = (255.0 * visual_obs)
        # else:
        #     visual_obs = (255.0 * visual_obs)
        if self._is_otc:
            v = {'visual':visual_obs, 'vector':self._get_vector_obs()}
            return v
        return visual_obs

class AddActionToVectorObs(gym.Wrapper):
    def __init__(self, env):
        super(AddActionToVectorObs, self).__init__(env)
        size = self.observation_space.spaces['vector'].shape[0] + self.action_space.n
        self.observation_space.spaces['vector'].shape = (size,)

    def _one_hot_encode(self, action):
        one_hot = np.zeros(self.action_space.n)
        one_hot[action] = 1.
        return one_hot

    def step(self, action):
        ob, rew, done, info = self.env.step(action)
        ob['vector'] = np.append(ob['vector'], self._one_hot_encode(action))
        return ob, rew, done, info
    
    def reset(self):
        ob = self.env.reset()
        ob['vector'] = np.append(ob['vector'], self._one_hot_encode(0))
        return ob   

class AddRewardToVectorObs(gym.Wrapper):
    def __init__(self, env):
        super(AddRewardToVectorObs, self).__init__(env)
        size = self.observation_space.spaces['vector'].shape[0] + 1
        self.observation_space.spaces['vector'].shape = (size,)

    def step(self, action):
        ob, rew, done, info = self.env.step(action)
        ob['vector'] = np.append(ob['vector'], rew)
        return ob, rew, done, info
    
    def reset(self):
        ob = self.env.reset()
        ob['vector'] = np.append(ob['vector'], 0)
        return ob   

class RemoveVectorObs(gym.Wrapper):
    def __init__(self, env):
        super(RemoveVectorObs, self).__init__(env)
        size = 0
        self.observation_space.spaces['vector'].shape = (size,)

    def step(self, action):
        ob, rew, done, info = self.env.step(action)
        v = np.zeros(0, dtype=np.float32)
        ob['vector'] = v
        return ob, rew, done, info
    
    def reset(self):
        ob = self.env.reset()
        v = np.zeros(0, dtype=np.float32)
        ob['vector'] = v
        return ob    

class NormalizeWrapper(gym.ObservationWrapper):
    def __init__(self, env, normal_filename = None):
        super(NormalizeWrapper, self).__init__(env)
        self._has_vector_obs = hasattr(self.observation_space, 'spaces')
        if self._has_vector_obs:
            self._visual_obs_shape = self.observation_space.spaces['visual'].shape
            is_8bit = self.observation_space.spaces['visual'] is np.uint8
        else:
            self._visual_obs_shape = self.observation_space.shape
            is_8bit = self.observation_space.dtype is np.uint8
        self._divBy255 = not is_8bit 
        self._divBy255 = True # HACK
        self.mean = None
        self.std = None
        if normal_filename:
            self.mean = np.loadtxt(normal_filename+'_mean.txt')
            self.mean = self.mean.reshape(self._visual_obs_shape)
            self.std = np.loadtxt(normal_filename+'_std.txt')

    def observation(self, obs):
        # x = (tf.to_float(x) - self.ob_mean) / self.ob_std
        visual_obs = obs['visual'] if self._has_vector_obs else obs
        if self.mean is not None:
            visual_obs = (visual_obs - self.mean) / self.std
        elif self._divBy255:
            visual_obs = visual_obs / 255
        if self._has_vector_obs:
            obs['visual'] = visual_obs
        else:
            obs = visual_obs
        return obs
            
            
class OTCPreprocessing(gym.Wrapper):
    """A class implementing image preprocessing for OTC agents.

    Specifically, this converts observations to greyscale. It doesn't
    do anything else to the environment.
    """

    def __init__(self, environment, args):
        """Constructor for an Obstacle Tower preprocessor.

        Args:
            environment: Gym environment whose observations are preprocessed.

        """
        self.env = environment

        environment.action_meanings = ['NOOP']
        # environment.np_random, seed = gym.utils.seeding.np_random(None)
        action_set_5 = {
            0: [0, 0, 0, 0],  # nop
            1: [1, 0, 0, 0],  # forward
            2: [0, 1, 0, 0],  # cam left
            3: [0, 2, 0, 0],  # cam right
            4: [1, 0, 1, 0],   # jump forward
        }
        action_set_6 = {
            0: [0, 0, 0, 0],  # nop
            1: [1, 0, 0, 0],  # forward
            2: [2, 0, 0, 0],  # backward
            3: [0, 1, 0, 0],  # cam left
            4: [0, 2, 0, 0],  # cam right
            5: [1, 0, 1, 0],   # jump forward
        }
        action_set_27_limit_jump = {
            0: [0, 0, 0, 0],  # nop
            1: [1, 0, 0, 0],  # forward
            2: [2, 0, 0, 0],  # backward
            3: [0, 1, 0, 0],  # cam left
            4: [0, 2, 0, 0],  # cam right
            5: [1, 1, 0, 0],  # forward + cam left
            6: [2, 1, 0, 0],  # backward + cam left
            7: [1, 2, 0, 0],  # forward + cam right
            8: [2, 2, 0, 0],  # backward + cam right
            9: [0, 0, 0, 1],  # left
            10: [0, 0, 0, 2],  # right
            11: [1, 0, 0, 1],  # left + forward
            12: [2, 0, 0, 1],  # left + backward
            13: [0, 1, 0, 1],  # left + cam left
            14: [0, 2, 0, 1],  # left + cam right
            15: [1, 1, 0, 1],  # left + forward + cam left
            16: [2, 1, 0, 1],  # left + backward + cam left
            17: [1, 2, 0, 1],  # left + forward + cam right
            18: [2, 2, 0, 1],  # left + backward + cam right
            19: [1, 0, 0, 2],  # right + forward
            20: [2, 0, 0, 2],  # right + backward
            21: [0, 1, 0, 2],  # right + cam left
            22: [0, 2, 0, 2],  # right + cam right
            23: [1, 1, 0, 2],  # right + forward + cam left
            24: [2, 1, 0, 2],  # right + backward + cam left
            25: [1, 2, 0, 2],  # right + forward + cam right
            26: [2, 2, 0, 2],  # right + backward + cam right
            27: [1, 0, 1, 0],  # jump + forward
        }
        action_set_20_limit_backwards_and_jump = {
            0: [0, 0, 0, 0],  # nop
            1: [1, 0, 0, 0],  # forward
            2: [2, 0, 0, 0],  # backward
            3: [0, 1, 0, 0],  # cam left
            4: [0, 2, 0, 0],  # cam right
            5: [1, 1, 0, 0],  # forward + cam left
            6: [1, 2, 0, 0],  # forward + cam right
            7: [0, 0, 0, 1],  # left
            8: [0, 0, 0, 2],  # right
            9: [1, 0, 0, 1],  # left + forward
            10: [0, 1, 0, 1],  # left + cam left
            11: [0, 2, 0, 1],  # left + cam right
            12: [1, 1, 0, 1],  # left + forward + cam left
            13: [1, 2, 0, 1],  # left + forward + cam right
            14: [1, 0, 0, 2],  # right + forward
            15: [0, 1, 0, 2],  # right + cam left
            16: [0, 2, 0, 2],  # right + cam right
            17: [1, 1, 0, 2],  # right + forward + cam left
            18: [1, 2, 0, 2],  # right + forward + cam right
            19: [1, 0, 1, 0],  # jump + forward
        }
        action_set_54 = {
            0: [0, 0, 0, 0],  # nop
            1: [1, 0, 0, 0],  # forward
            2: [2, 0, 0, 0],  # backward
            3: [0, 1, 0, 0],  # cam left
            4: [0, 2, 0, 0],  # cam right
            5: [1, 1, 0, 0],  # forward + cam left
            6: [2, 1, 0, 0],  # backward + cam left
            7: [1, 2, 0, 0],  # forward + cam right
            8: [2, 2, 0, 0],  # backward + cam right
            9: [0, 0, 0, 1],  # left
            10: [0, 0, 0, 2],  # right
            11: [1, 0, 0, 1],  # left + forward
            12: [2, 0, 0, 1],  # left + backward
            13: [0, 1, 0, 1],  # left + cam left
            14: [0, 2, 0, 1],  # left + cam right
            15: [1, 1, 0, 1],  # left + forward + cam left
            16: [2, 1, 0, 1],  # left + backward + cam left
            17: [1, 2, 0, 1],  # left + forward + cam right
            18: [2, 2, 0, 1],  # left + backward + cam right
            19: [1, 0, 0, 2],  # right + forward
            20: [2, 0, 0, 2],  # right + backward
            21: [0, 1, 0, 2],  # right + cam left
            22: [0, 2, 0, 2],  # right + cam right
            23: [1, 1, 0, 2],  # right + forward + cam left
            24: [2, 1, 0, 2],  # right + backward + cam left
            25: [1, 2, 0, 2],  # right + forward + cam right
            26: [2, 2, 0, 2],  # right + backward + cam right
            27: [0, 0, 1, 0],  # jump 
            28: [1, 0, 1, 0],  # jump + forward
            29: [2, 0, 1, 0],  # jump + backward
            30: [0, 1, 1, 0],  # jump + cam left
            31: [0, 2, 1, 0],  # jump + cam right
            32: [1, 1, 1, 0],  # jump + forward + cam left
            33: [2, 1, 1, 0],  # jump + backward + cam left
            34: [1, 2, 1, 0],  # jump + forward + cam right
            35: [2, 2, 1, 0],  # jump + backward + cam right
            36: [0, 0, 1, 1],  # jump + left
            37: [0, 0, 1, 2],  # jump + right
            38: [1, 0, 1, 1],  # jump + left + forward
            39: [2, 0, 1, 1],  # jump + left + backward
            40: [0, 1, 1, 1],  # jump + left + cam left
            41: [0, 2, 1, 1],  # jump + left + cam right
            42: [1, 1, 1, 1],  # jump + left + forward + cam left
            43: [2, 1, 1, 1],  # jump + left + backward + cam left
            44: [1, 2, 1, 1],  # jump + left + forward + cam right
            45: [2, 2, 1, 1],  # jump + left + backward + cam right
            46: [1, 0, 1, 2],  # jump + right + forward
            47: [2, 0, 1, 2],  # jump + right + backward
            48: [0, 1, 1, 2],  # jump + right + cam left
            49: [0, 2, 1, 2],  # jump + right + cam right
            50: [1, 1, 1, 2],  # jump + right + forward + cam left
            51: [2, 1, 1, 2],  # jump + right + backward + cam left
            52: [1, 2, 1, 2],  # jump + right + forward + cam right
            53: [2, 2, 1, 2],  # jump + right + backward + cam rght
        }

        custom_action_set = action_set_5 if args.action_set_5 else None
        custom_action_set = action_set_6 if args.action_set_6 else custom_action_set
        custom_action_set = action_set_20_limit_backwards_and_jump if args.action_set_20 else custom_action_set
        custom_action_set = action_set_27_limit_jump if args.action_set_27 else custom_action_set
        custom_action_set = action_set_54 if args.action_set_54 else custom_action_set
        self.game_over = False
        self.lives = 0  # Will need to be set by reset().
        if custom_action_set:
            self._action_lookup = custom_action_set
        else:
            self._action_lookup = {
                0: [0, 0, 0, 0],  # nop
                1: [1, 0, 0, 0],  # forward
                2: [2, 0, 0, 0],  # backward
                3: [0, 1, 0, 0],  # cam left
                4: [0, 2, 0, 0],  # cam right
                5: [1, 0, 1, 0],   # jump forward
                6: [1, 1, 0, 0],  # forward + cam left
                7: [1, 2, 0, 0]  # forward + cam right
            }

    # @property
    # def _action_lookup(self):
    #     return  {
    #         0: [0, 0, 0, 0],  # nop
    #         1: [1, 0, 0, 0],  # forward
    #         2: [2, 0, 0, 0],  # backward
    #         3: [0, 1, 0, 0],  # cam left
    #         4: [0, 2, 0, 0],  # cam right
    #         5: [1, 0, 1, 0],   # jump forward
    #         6: [1, 1, 0, 0],  # forward + cam left
    #         7: [1, 2, 0, 0],  # forward + cam right
    #     }

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return spaces.Discrete(len(self._action_lookup))
        # return self.env.action_space

    @property
    def reward_range(self):
        return self.env.reward_range

    @property
    def metadata(self):
        return self.env.metadata

    def reset(self):
        """Resets the environment. Converts the observation to greyscale, 
        if it is not. 

        Returns:
        observation: numpy array, the initial observation emitted by the
            environment.
        """
        observation = self.env.reset()
        # if(len(observation.shape)> 2):
        #     observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)

        return observation

    def render(self, mode):
        """Renders the current screen, before preprocessing.

        This calls the Gym API's render() method.

        Args:
        mode: Mode argument for the environment's render() method.
            Valid values (str) are:
            'rgb_array': returns the raw ALE image.
            'human': renders to display via the Gym renderer.

        Returns:
        if mode='rgb_array': numpy array, the most recent screen.
        if mode='human': bool, whether the rendering was successful.
        """
        return self.env.render(mode)

    def step(self, actionInput):
        """Applies the given action in the environment. Converts the observation to 
        greyscale, if it is not. 

        Remarks:

        * If a terminal state (from life loss or episode end) is reached, this may
            execute fewer than self.frame_skip steps in the environment.
        * Furthermore, in this case the returned observation may not contain valid
            image data and should be ignored.

        Args:
        action: The action to be executed.

        Returns:
        observation: numpy array, the observation following the action.
        reward: float, the reward following the action.
        is_terminal: bool, whether the environment has reached a terminal state.
            This is true when a life is lost and terminal_on_life_loss, or when the
            episode is over.
        info: Gym API's info data structure.
        """
        # ['Movement Forward/Back', 'Camera', 'Jump', 'Movement Left/Right']
        # [3, 3, 2, 3]
        from typing import Iterable
        if isinstance(actionInput, Iterable):
            action = self._action_lookup[actionInput[0]]
        else:
            action = self._action_lookup[actionInput]

        observation, reward, game_over, info = self.env.step(action)
        self.game_over = game_over
        # if(len(observation.shape)> 2):
        #     observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        info['actual_action']=actionInput
        info['actual_inner_action']=action
        return observation, reward, game_over, info

    def unwrap(self):
        if hasattr(self.env, "unwrapped"):
            return env.unwrapped
        elif hasattr(self.env, "env"):
            return unwrap(self.env.env)
        elif hasattr(self.env, "leg_env"):
            return unwrap(self.env.leg_env)
        else:
            return self.env
