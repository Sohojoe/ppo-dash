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
from a2c_ppo_acktr.envs import VecNormalize

class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs, ) +
                                       low.shape).to(device)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:]
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()

class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self._vector_obs_len = 0
        self.device = device
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
        vector_obs = np.zeros(0,np.float32)
        if self._has_vector_obs:
            vector_obs = obs['vector']
            obs = obs['visual']
        obs = torch.from_numpy(obs).float().to(self.device)
        vector_obs = torch.from_numpy(vector_obs).float().to(self.device)
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

def make_env_all_params(rank, add_monitor, args, sleep_multiple=1):
    from time import sleep
    # cudaDevice = int(os.environ['CUDA_VISIBLE_DEVICES'])
    cudaDevice = -1
    sleep_time = rank
    if cudaDevice is not -1:
        envs_per_process = args.envs_per_process
        # sleep_time = rank + (cudaDevice)
        sleep_time = rank + (cudaDevice * envs_per_process)
        rank = rank + (cudaDevice * envs_per_process)
    if args.score:
        sleep_time = 0
    is_grading = os.getenv('OTC_EVALUATION_ENABLED', False)
    if is_grading:
        rank = 0
        sleep_time = 0
    # sleep(sleep_multiple * sleep_time)
    sleep(1 * sleep_time)
    if rank >= 35:
        rank += 1
    from sohojoe_obstacle_tower_env import ObstacleTowerEnv
    from otc_preprocessing import OTCPreprocessing
    environment_path = args.environment_filename
    assert environment_path is not None
    real_time = args.real_time
    show_obs = rank == 1 or rank == 0 or args.score
    if is_grading:
        show_obs = False
    if args.env == 'ObtRetro-v4':
        env = ObstacleTowerEnv(
            environment_path,
            worker_id=rank,
            timeout_wait=6000,
            retro=False,
            docker_training=args.docker_training,
            realtime_mode=real_time)
        if not is_grading and not args.sample_normal:
            if args.score:
                env = ScoreLevelWrapper(env)
            # elif show_obs and args['seed_from_10']:
            elif args.seed_from_10:
                env = TrainLevelWrapper(env, 'seed_from_10')
            elif args.level_is_rand:
                env = TrainLevelWrapper(env, 'rand')
            else:
                env = TrainLevelWrapper(env)
        env = RetroWrapper(env, args.sample_normal)
        env = OTCPreprocessing(env)
        # env = ProcessFrame84Color(env)
        if show_obs:
            env = RenderObservations(env)
            env = KeyboardControlWrapper(env)
        env = FrameStackMono(env, 2)
    elif args.env == 'ObtRetro-v6':
        env = ObstacleTowerEnv(
            environment_path,
            worker_id=rank,
            timeout_wait=6000,
            retro=False,
            docker_training=args.docker_training,
            realtime_mode=real_time)
        if not is_grading and not args.sample_normal:
            if args.score:
                env = ScoreLevelWrapper(env)
            # elif show_obs and args['seed_from_10']:
            elif args.seed_from_10:
                env = TrainLevelWrapper(env, 'seed_from_10')
            elif args.level_is_rand:
                env = TrainLevelWrapper(env, 'rand')
            else:
                env = TrainLevelWrapper(env)
        env = RetroWrapper(env, args.sample_normal)
        env = OTCPreprocessing(env)
        # env = ProcessFrame84Color(env)
        if show_obs:
            env = RenderObservations(env)
            env = KeyboardControlWrapper(env)
    elif args.env == 'ObtRetro-v7':
        env = ObstacleTowerEnv(
            environment_path,
            worker_id=rank,
            timeout_wait=6000,
            retro=False,
            docker_training=args.docker_training,
            realtime_mode=real_time)
        if not is_grading and not args.sample_normal:
            if args.score:
                env = ScoreLevelWrapper(env)
            # elif show_obs and args['seed_from_10']:
            elif args.seed_from_10:
            # elif args.seed_from_10 and show_obs:
                env = TrainLevelWrapper(env, 'seed_from_10', show_obs)
            elif args.level_is_rand:
                env = TrainLevelWrapper(env, 'rand', show_obs)
            else:
                # env = TrainLevelWrapper(env, show_debug=show_obs)
                env = TrainLevelWrapper(env)
        env = RetroWrapper(env, args.sample_normal, keep_obs=True)
        env = OTCPreprocessing(env)
        if show_obs:
            env = RenderObservations(env)
            env = KeyboardControlWrapper(env)
        env = AddActionToVectorObs(env)
        env = NormalizeWrapper(env, "ObtRetro-v6")
        env = FrameStackMono(env, 2)    

    # 
    # wrap for PyTorch convolutions
    # obs_shape = env.observation_space.shape
    # if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
    env = TransposeImage(env, op=[2, 0, 1])

    # if add_monitor:
    #     logdir = osp.join('summaries', args.exp_name)
    #     logger.configure(logdir)
    #     # env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
    #     env = Monitor(
    #         env,
    #         osp.join(logger.get_dir(), '%.2i' % rank),
    #         allow_early_resets=True)


    return env


def make_otc_env(args,
                 device,
                 start_index=0,
                 allow_early_resets=True,
                 start_method=None):

    environment_path = args.environment_filename

    def make_env(rank):
        def _thunk():
            env = make_env_all_params(rank, True, args)
            # env.seed(seed + rank)
            # env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
            #               allow_early_resets=allow_early_resets)
            # return wrap_deepmind(env, **wrapper_kwargs)
            return env

        return _thunk

    envs = [
        make_env(i + start_index + 1)
        for i in range(args.num_processes)
    ]

    if args.num_processes == 1:
        envs = DummyVecEnv(envs)
    else:
        envs = ShmemVecEnv(envs)
    # if args.normalize_visual_obs:
    #     envs = VecNormalize(envs, True, ret=False,clipob=1.)
    return envs


def otc_arg_parser():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--algo', default='ppo', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument(
        '--gail',
        action='store_true',
        default=False,
        help='do imitation learning with gail')
    parser.add_argument(
        '--gail-experts-dir',
        default='./gail_experts',
        help='directory that contains expert demonstrations for gail')
    parser.add_argument(
        '--gail-batch-size',
        type=int,
        default=128,
        help='gail batch size (default: 128)')
    parser.add_argument(
        '--gail-epoch', type=int, default=5, help='gail epochs (default: 5)')
    parser.add_argument(
        '--lr', 
        type=float, 
        # default=7e-4, 
        # default=2.5e-4, 
        default=1e-4, 
        help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        # default=0.96,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=True, # False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        # default=0.01,
        default=0.001,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default =0.5,
        # default=1.,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        # default=16,
        default=32, # 16,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-steps',
        type=int,
        # default=5,
        default=512,
        # default=256,
        # default=128,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        # default=4,
        default=8,
        # default=3,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=32,
        #=8,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param',
        type=float,
        # default=0.2,
        default=0.1,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=1,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=10,
        # default=100,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=None,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=10e6,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--env',
        default='ObtRetro-v6', #'PongNoFrameskip-v4',
        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument(
        '--log-dir',
        default='./summaries',
        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument(
        '--save-dir',
        default='./models/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=True, # False,
        # default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        # default=True,
        help='use a linear schedule on the learning rate')

    parser.add_argument(
        '--policy',
        help='Policy architecture',
        choices=['cnn', 'lstm', 'lnlstm', 'mlp'],
        default='lnlstm')

    # parser.add_argument('--lbda', type=float, default=0.95)
    # parser.add_argument('--gamma', type=float, default=0.96)  # 0.99
    # parser.add_argument('--nminibatches', type=int, default=8)
    # parser.add_argument('--norm_adv', type=int, default=1)
    # parser.add_argument('--norm_rew', type=int, default=0)
    # parser.add_argument(
    #     '--lr', type=float, default=1e-4)  # lambda f: f * 2.5e-4,
    # parser.add_argument('--ent_coeff', type=float, default=0.001)
    # parser.add_argument('--nepochs', type=int, default=8)
    parser.add_argument('--nsteps_per_seg', type=int, default=512)
    # parser.add_argument('--num_timesteps', type=int, default=int(1e7))

    # parser.add_argument('--env', help='environment ID', default='ObtRetro-v4')
    # parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument(
        'environment_filename',
        default='../ObstacleTower/obstacletower',
        nargs='?')
    # parser.add_argument('--envs_per_process', type=int, default=8)
    parser.add_argument('--real_time', action='store_true')
    parser.add_argument('--score', type=bool, default=False)
    parser.add_argument('--docker_training', action='store_true')
    parser.set_defaults(docker_training=False)
    parser.add_argument('--sample_normal', action='store_true')
    parser.add_argument('--seed_from_10', action='store_true')
    parser.add_argument('--level_is_rand', action='store_true')
    parser.add_argument('--exp_name', type=str, default='debug')
    # parser.add_argument('--save_freq', type=int, default=50000)
    parser.add_argument('--load', action='store_true')
    # parser.add_argument('--normalize_visual_obs', action='store_true')

    return parser
