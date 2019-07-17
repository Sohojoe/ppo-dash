import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
# from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
from make_env import make_otc_env, otc_arg_parser
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize
from sohojoe_wrappers import done_grading, is_grading

def main():
    parser = otc_arg_parser()
    # args = get_args()
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.score = True
    args.load = True
    args.num_processes = 1

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    # utils.cleanup_log_dir(log_dir)
    # utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    device = torch.device("cpu")

    # envs = make_vec_envs(args.env, args.seed, args.num_processes,
    #                      args.gamma, args.log_dir, device, False)
    envs = make_otc_env(args, device, start_index=258)

    save_path = os.path.join(args.save_dir, args.exp_name)
    actor_critic, ob_rms = \
            torch.load(
                os.path.join(save_path, args.env + ".pt"),
                map_location='cpu')
    actor_critic.to(device)
    from make_env import VecPyTorch,  VecPyTorchFrameStack
    envs = VecPyTorch(envs, device)
    # envs = VecPyTorchFrameStack(envs, 1, device)
    env = envs #[0]

    vec_norm = get_vec_normalize(env)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    recurrent_hidden_states = torch.zeros(1,
                                        actor_critic.recurrent_hidden_state_size)
    masks = torch.zeros(1, 1)

    obs, vector_obs = env.reset()

    episode_reward = 0
    episode_rewards = []
    total_episodes = 0
    max_level = 0
    max_levels = []

    while True:
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(
                obs,
                vector_obs,
                recurrent_hidden_states, 
                masks, 
                deterministic=args.cuda_deterministic)

        # Obser reward and next obs
        obs, vector_obs, reward, done, _ = env.step(action)

        masks.fill_(0.0 if done else 1.0)
        reward = float(reward.mean())
        episode_reward += reward
        if reward == 1:
            max_level += 1
        # elif reward > 0:
        #     print ('reward:', reward)
        if done:
            episode_rewards.append(episode_reward)
            ave_reward = sum(episode_rewards) / len(episode_rewards)
            total_episodes +=1
            max_levels.append(max_level)
            ave_level = sum(max_levels) / len(max_levels)
            print ('ep:', total_episodes, 'level:', max_level, 'ave_level:', round(ave_level,2), 'episode_reward:', round(episode_reward,2), 'ave_reward', round(ave_reward,2))
            episode_reward = 0
            max_level = 0
            if is_grading(env):
                if done_grading(env):
                    break
            elif total_episodes >= 25:
                break
            # obs = env.reset()


if __name__ == "__main__":
    main()

