import os
import gym
import numpy as np
from collections import deque
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)
from baselines.common.atari_wrappers import LazyFrames


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
        # self._levels = [0,1,2,3,4,5,6]
        self._runAchievedLevel = np.zeros((5,5))
        self._runAchievedScore = np.zeros((5,5))
        self._seedIdx = 0
        self._runIdx = 0
        self._level = 0
        self._max_level = 0
        self._ep_rew = 0
        print ('score. Seeds', [n for n in self._scoreSeeds])
        self._setLevel()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self._ep_rew += reward
        if reward >= 1:
            self._level += 1

        if done:
            # out of time
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
            print (
                'seed:', self._scoreSeeds[i], 
                'ave level:', np.average(self._runAchievedLevel[i,:]), self._runAchievedLevel[i,:], 
                'ave score:', np.average(self._runAchievedScore[i,:]), self._runAchievedScore[i,:])
        print ('total ave level:', np.average(self._runAchievedLevel), 'total ave score:', np.average(self._runAchievedScore))


class TrainLevelWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        # np.random.uniform generates uniformly
        seeds = [x for x in range(100)]
        # np.random.shuffle(seeds)
        self._trainSeeds = seeds[0:95]
        self._level = 0
        self._max_level = 0
        print ('train. Seeds', [n for n in self._trainSeeds])
        self._setLevel()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        if done:
            # out of time
            reward = -1
            self._level = 0
            # if np.random.randint(0,2) == 0:
            #     self._level = 0

        # +0.1 is provided for opening doors, solving puzzles, or picking up keys.
        if reward >= 1:
            done = True
            self._level += 1
            if self._level > self._max_level:
                self._max_level = self._level
                print ('new max level:', self._max_level)
        elif reward > 0:
            reward = 1
        return ob, reward, done, info   

    def reset(self):
        self._setLevel()
        return self.env.reset()
    
    def _setLevel(self):
        self.env.floor(self._level)
        self.env.dense_reward = 1
        self._last_seed = np.random.choice(self._trainSeeds)
        self.env.seed(self._last_seed)

# class ReduceWrapper(gym.Wrapper):
#     def __init__(self, env):
#         gym.Wrapper.__init__(self, env)
#         env.action_meanings = ['NOOP']
#         # env.np_random, seed = gym.utils.seeding.np_random(None)

#         self.game_over = False
#         self.lives = 0  # Will need to be set by reset().
#         self._action_lookup = {
#             0: [0, 0, 0, 0],  # nop
#             1: [1, 0, 0, 0],  # forward
#             2: [2, 0, 0, 0],  # backward
#             3: [0, 1, 0, 0],  # cam left
#             4: [0, 2, 0, 0],  # cam right
#             5: [1, 0, 1, 0],   # jump forward
#             6: [1, 1, 0, 0],  # forward + cam left
#             7: [1, 2, 0, 0]  # forward + cam right
#         }
#     @property
#     def action_space(self):
#         return spaces.Discrete(len(self._action_lookup))


class RenderWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        
    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.env.render()
        return ob, reward, done, info   

class RenderObservations(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.viewer = None
        
    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self._renderObs(ob)
        return ob, reward, done, info   

    def _renderObs(self, img):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()
        self.viewer.imshow(img)
        return self.viewer.isopen

    def close(self):
        self.env.close()
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class ProcessFrame84Color(gym.ObservationWrapper):
    def __init__(self, env):
        super(ProcessFrame84Color, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84Color.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 84 * 84 * 3:  # otc
            img = np.reshape(frame, [84, 84, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution." + str(frame.size)
        return img.astype(np.uint8)

class ProcessFrame168Color(gym.ObservationWrapper):
    def __init__(self, env):
        super(ProcessFrame168Color, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(168, 168, 3), dtype=np.uint8)

    def observation(self, obs):
        frame = ProcessFrame168Color.process(obs[0])
        return frame

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._frame = self.observation(observation)
        return self._frame, reward, done, info

    @staticmethod
    def process(frame):
        if frame.size == 168 * 168 * 3:  # otc
            img = frame * 255.0
        else:
            assert False, "Unknown resolution." + str(frame.size)
        return img.astype(np.uint8)

class ColorRandomization(gym.ObservationWrapper):
    def __init__(self, env):
        super(ColorRandomization, self).__init__(env)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def observation(self, obs):
        frame = self.randomize_image(obs)
        return frame

    def randomize_image(self, img):
        # rndBright = np.random.randint(0,255)
        # rndSat = np.random.randint(0,255)
        rndBright = np.random.randint(-128,128)
        rndSat = np.random.randint(-128,128)
        rndHue = np.random.randint(-8,8)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        # lim = 255 - rndBright
        # v[v > lim] = 255
        # v[v <= lim] += rndBright
        # sat = 255 - rndSat
        # s[s > sat] = 255
        # s[s <= sat] += rndSat
        # hue = 255 - rndHue
        # s[s > hue] = 255
        # s[s <= hue] += rndHue
        # lim = 255 - rndBright
        # v[v > lim] = 255
        # v[v <= lim] += rndBright

        # v = cv2.equalizeHist(v)
        # s = cv2.equalizeHist(s)
        h += np.uint8(rndHue)
        if rndSat >0:
            rndSat = np.uint8(rndSat)
            sat = 255 - rndSat
            s[s > sat] = 255
            s[s <= sat] += rndSat
        elif rndSat < 0:
            rndSat = np.uint8(0-rndSat)
            s[s < rndSat] = 0
            s[s >= rndSat] += rndSat
        if rndBright >0:
            rndBright = np.uint8(rndBright)
            sat = 255 - rndBright
            s[s > sat] = 255
            s[s <= sat] += rndBright
        elif rndBright < 0:
            rndBright = np.uint8(0-rndBright)
            s[s < rndBright] = 0
            s[s >= rndBright] += rndBright
        # s += np.uint8(rndSat)
        # v += np.uint8(rndBright)

        # v[v >64] = 64
        # s[s <160] = 0

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
        return img
    def increase_brightness(self, img, value=30):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
        return img

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
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] + k-1,)), dtype=env.observation_space.dtype)

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
        for i in range(self.k-1):
            self.frames.append(self.mono_frames[i+1])
        assert len(self.frames) == self.observation_space.shape[2]
        # output = LazyFrames(list(self.frames))
        output = np.array(list(self.frames)).T
        return output

class RetroWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(RetroWrapper, self).__init__(env)

    def observation(self, obs):
        frame = (255.0 * obs).astype(np.uint8)
        return frame
