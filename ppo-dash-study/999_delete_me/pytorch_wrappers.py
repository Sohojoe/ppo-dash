import os
from sohojoe_wrappers import *
import numpy as np
import os.path as osp
import torch
from gym.spaces.box import Box
from sohojoe_wrappers import done_grading, is_grading
from baselines.common.vec_env import VecEnvWrapper
# from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
# from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from sohojoe_dummy_vec_env import DummyVecEnv
from sohojoe_shmem_vec_env import ShmemVecEnv
from ppo.envs import VecNormalize


# class VecPyTorchFrameStack(VecEnvWrapper):
#     def __init__(self, venv, nstack, device=None):
#         self.venv = venv
#         self.nstack = nstack

#         wos = venv.observation_space  # wrapped ob space
#         self.shape_dim0 = wos.shape[0]

#         low = np.repeat(wos.low, self.nstack, axis=0)
#         high = np.repeat(wos.high, self.nstack, axis=0)

#         if device is None:
#             device = torch.device('cpu')
#         self.stacked_obs = torch.zeros((venv.num_envs, ) +
#                                        low.shape).to(device)

#         observation_space = gym.spaces.Box(
#             low=low, high=high, dtype=venv.observation_space.dtype)
#         VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

#     def step_wait(self):
#         obs, rews, news, infos = self.venv.step_wait()
#         self.stacked_obs[:, :-self.shape_dim0] = \
#             self.stacked_obs[:, self.shape_dim0:]
#         for (i, new) in enumerate(news):
#             if new:
#                 self.stacked_obs[i] = 0
#         self.stacked_obs[:, -self.shape_dim0:] = obs
#         return self.stacked_obs, rews, news, infos

#     def reset(self):
#         obs = self.venv.reset()
#         if torch.backends.cudnn.deterministic:
#             self.stacked_obs = torch.zeros(self.stacked_obs.shape)
#         else:
#             self.stacked_obs.zero_()
#         self.stacked_obs[:, -self.shape_dim0:] = obs
#         return self.stacked_obs

#     def close(self):
#         self.venv.close()

class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device, half_precision = False):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self._vector_obs_len = 0
        self.device = device
        self._half_precision = half_precision
        self._has_vector_obs = hasattr(self.observation_space, 'spaces')
        if self._has_vector_obs:
            self._vector_obs_len = self.observation_space.spaces['vector'].shape[0]
            self.observation_space = self.observation_space.spaces['visual']
        # TODO: Fix data types

    @property
    def vector_obs_len(self):
        return self._vector_obs_len

    def reset(self):
        vector_obs = np.zeros(0,np.float32)
        obs = self.venv.reset()
        if self._has_vector_obs:
            vector_obs = obs['vector']
            obs = obs['visual']
        if self._half_precision:
            obs = torch.from_numpy(obs).half().to(self.device)
            vector_obs = torch.from_numpy(vector_obs).half().to(self.device)
        else:
            obs = torch.from_numpy(obs).float().to(self.device)
            vector_obs = torch.from_numpy(vector_obs).float().to(self.device)
        return obs, vector_obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        if self._half_precision:
            vector_obs = np.zeros(0,np.float16)
            if self._has_vector_obs:
                vector_obs = obs['vector']
                obs = obs['visual']
            obs = torch.from_numpy(obs).half().to(self.device, non_blocking=True)
            vector_obs = torch.from_numpy(vector_obs).half().to(self.device, non_blocking=True)
            reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        else:
            vector_obs = np.zeros(0,np.float32)
            if self._has_vector_obs:
                vector_obs = obs['vector']
                obs = obs['visual']
            obs = torch.from_numpy(obs).float().to(self.device, non_blocking=True)
            vector_obs = torch.from_numpy(vector_obs).float().to(self.device, non_blocking=True)
            reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, vector_obs, reward, done, info

class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)

class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, f"Error: Operation, {str(op)}, must be dim3"
        self.op = op
        self._has_vector_obs = hasattr(self.observation_space, 'spaces')
        if self._has_vector_obs:
            obs_space = self.observation_space.spaces['visual']
            obs_shape = obs_space.shape
            self._full_observation_space = self.observation_space
            image_space = Box(
                obs_space.low[0, 0, 0],
                obs_space.high[0, 0, 0], [
                    obs_shape[self.op[0]], obs_shape[self.op[1]],
                    obs_shape[self.op[2]]
                ],
                dtype=obs_space.dtype)
            self.observation_space.spaces['visual'] = image_space
        else:
            obs_shape = self.observation_space.shape
            self.observation_space = Box(
                self.observation_space.low[0, 0, 0],
                self.observation_space.high[0, 0, 0], [
                    obs_shape[self.op[0]], obs_shape[self.op[1]],
                    obs_shape[self.op[2]]
                ],
                dtype=self.observation_space.dtype)
        
    def observation(self, ob):
        if self._has_vector_obs:
            visual_obs = ob['visual'].transpose(self.op[0], self.op[1], self.op[2])
            ob['visual']=visual_obs
            return ob
        return ob.transpose(self.op[0], self.op[1], self.op[2])