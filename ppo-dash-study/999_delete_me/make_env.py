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
# from inverse_rl import InverseRL
from pytorch_wrappers import TransposeImage

def make_env_all_params(rank, env_count, total_env, add_monitor, args, sleep_multiple=1):
    from time import sleep
    # cudaDevice = int(os.environ['CUDA_VISIBLE_DEVICES'])
    cudaDevice = -1
    sleep_time = rank
    is_pratice_agent = False
    if args.num_practice_agents < 0 and env_count < (-args.num_practice_agents):
        is_pratice_agent = True
    if args.num_practice_agents > 0 and env_count >= (total_env-args.num_practice_agents):
        is_pratice_agent = True
    print ('is_pratice_agent', is_pratice_agent, 'env_count', env_count)
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
    from sohojoe_wrappers import OTCPreprocessing
    environment_path = args.environment_filename
    assert environment_path is not None
    real_time = args.real_time
    show_obs = rank == 1 or rank == 0 or args.score
    if is_grading:
        show_obs = False
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
                env = TrainLevelWrapper(env, env_count, 'seed_from_10')
            elif args.level_is_rand:
                env = TrainLevelWrapper(env, env_count, 'rand')
            elif args.alfie:
                env = TrainLevelWrapper(env, env_count, 'alfie')
            else:
                env = TrainLevelWrapper(env, env_count)
        env = RetroWrapper(env, args.sample_normal)
        env = OTCPreprocessing(env, custom_action_set)
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
                env = TrainLevelWrapper(env, env_count, 'seed_from_10')
            elif args.level_is_rand:
                env = TrainLevelWrapper(env, env_count, 'rand')
            elif args.alfie:
                env = TrainLevelWrapper(env, env_count, 'alfie')
            else:
                env = TrainLevelWrapper(env, env_count)
        env = RetroWrapper(env, args.sample_normal)
        env = OTCPreprocessing(env, custom_action_set)
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
                env = TrainLevelWrapper(env, env_count, 'seed_from_10', show_obs)
            elif is_pratice_agent:
                env = TrainLevelWrapper(env, env_count, 'seed_from_10', show_obs)
            elif args.level_is_rand:
                env = TrainLevelWrapper(env, env_count, 'rand', show_obs)
            elif args.alfie:
                env = TrainLevelWrapper(env, env_count, 'alfie', show_obs)
            else:
                # env = TrainLevelWrapper(env, env_count, show_debug=show_obs)
                env = TrainLevelWrapper(env, env_count)
        env = RetroWrapper(env, args.sample_normal, keep_obs=True)
        env = OTCPreprocessing(env, custom_action_set)
        if show_obs:
            env = RenderObservations(env)
            env = KeyboardControlWrapper(env)
        # if not is_grading and not args.sample_normal and not args.score:            
        #     if args.inverse_rl:
        #         env = InverseRL(env, './video')
        env = AddActionToVectorObs(env)
        env = NormalizeWrapper(env, "ObtRetro-v6")
        env = FrameStackMono(env, 2)    
    elif args.env == 'ObtRetro-v8':
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
                env = TrainLevelWrapper(env, env_count, 'seed_from_10', show_obs)
            elif is_pratice_agent:
                env = TrainLevelWrapper(env, env_count, 'seed_from_10', show_obs)
            elif args.level_is_rand:
                env = TrainLevelWrapper(env, env_count, 'rand', show_obs)
            elif args.alfie:
                env = TrainLevelWrapper(env, env_count, 'alfie', show_obs)
            else:
                # env = TrainLevelWrapper(env, env_count, show_debug=show_obs)
                env = TrainLevelWrapper(env, env_count)
        env = RetroWrapper(env, args.sample_normal, keep_obs=True)
        env = OTCPreprocessing(env, custom_action_set)
        if show_obs:
            env = RenderObservations(env)
            env = KeyboardControlWrapper(env)
        # if not is_grading and not args.sample_normal and not args.score:            
        #     if args.inverse_rl:
        #         env = InverseRL(env, './video')
        env = AddActionToVectorObs(env)
        env = NormalizeWrapper(env, "ObtRetro-v6")
        # env = FrameStackMono(env, 2)    
    elif args.env == 'ObtRetro-v8-no-norm':
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
                env = TrainLevelWrapper(env, env_count, 'seed_from_10', show_obs)
            elif is_pratice_agent:
                env = TrainLevelWrapper(env, env_count, 'seed_from_10', show_obs)
            elif args.level_is_rand:
                env = TrainLevelWrapper(env, env_count, 'rand', show_obs)
            elif args.alfie:
                env = TrainLevelWrapper(env, env_count, 'alfie', show_obs)
            else:
                # env = TrainLevelWrapper(env, env_count, show_debug=show_obs)
                env = TrainLevelWrapper(env, env_count)
        env = RetroWrapper(env, args.sample_normal, keep_obs=True)
        env = OTCPreprocessing(env, custom_action_set)
        if show_obs:
            env = RenderObservations(env)
            env = KeyboardControlWrapper(env)
        # if not is_grading and not args.sample_normal and not args.score:            
        #     if args.inverse_rl:
        #         env = InverseRL(env, './video')
        env = NormalizeWrapper(env)
        env = AddActionToVectorObs(env)



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

    def make_env(rank, env_count, total_env):
        def _thunk():
            env = make_env_all_params(rank, env_count, total_env, True, args)
            # env.seed(seed + rank)
            # env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
            #               allow_early_resets=allow_early_resets)
            # return wrap_deepmind(env, **wrapper_kwargs)
            return env

        return _thunk

    envs = [
        make_env(i + start_index + 1, i, args.num_processes)
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
    parser.add_argument('--num_practice_agents', type=int, default=0)
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
    parser.add_argument('--alfie', action='store_true')
    parser.add_argument('--level_is_rand', action='store_true')
    parser.add_argument('--exp_name', type=str, default='debug')
    # parser.add_argument('--save_freq', type=int, default=50000)
    parser.add_argument('--load', action='store_true')
    # parser.add_argument('--normalize_visual_obs', action='store_true')
    parser.add_argument('--inverse_rl', action='store_true')
    parser.add_argument('--action_set_5', action='store_true')
    parser.add_argument('--action_set_6', action='store_true')
    parser.add_argument('--action_set_20', action='store_true')
    parser.add_argument('--action_set_27', action='store_true')
    parser.add_argument('--action_set_54', action='store_true')
    parser.add_argument('--half_precision', action='store_true')

    return parser
