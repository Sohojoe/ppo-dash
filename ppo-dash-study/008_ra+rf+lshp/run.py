import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ppo import algo, utils
from ppo.arguments import get_args
# from ppo.envs import make_vec_envs
from ppo.model import Policy, CNNBase
from ppo.storage import RolloutStorage
from evaluation import evaluate
from make_env import make_otc_env, otc_arg_parser
from ppo.utils import get_render_func, get_vec_normalize
from torch.utils.tensorboard import SummaryWriter

# @profile
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

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    tf_log_dir = os.path.join(log_dir, args.exp_name)
    if not os.path.exists(tf_log_dir):
        os.makedirs(tf_log_dir)
    writer = SummaryWriter(log_dir=tf_log_dir)
    eval_log_dir = log_dir + "_eval"
    # history_file = os.path.join(log_dir, args.exp_name+'.csv')

    torch.set_num_threads(1)
    # device = torch.device("cuda:0" if args.cuda else "cpu")
    device = torch.device("cuda" if args.cuda else "cpu")

    # envs = make_vec_envs(args.env, args.seed, args.num_processes,
    #                      args.gamma, args.log_dir, device, False)
    envs = make_otc_env(args, device)

    save_path = os.path.join(args.save_dir, args.exp_name)
    if args.load:
        actor_critic, ob_rms = \
                torch.load(
                    os.path.join(save_path, args.env + ".pt"))
        vec_norm = get_vec_normalize(envs)
        if vec_norm is not None:
            vec_norm.eval()
            vec_norm.ob_rms = ob_rms
    else:
        obs_shape = envs.observation_space.spaces['visual'].shape
        if args.env == 'ObtRetro-baseline':
            obs_shape = (obs_shape[0]*4, obs_shape[1], obs_shape[2])
        vector_obs_len = envs.observation_space.spaces['vector'].shape[0]
        actor_critic = Policy(
            obs_shape,
            envs.action_space,
            base = CNNBase,
            base_kwargs={'recurrent': args.recurrent_policy},
            vector_obs_len=vector_obs_len)
    if torch.cuda.device_count() > 1:
        actor_critic_parallel = nn.DataParallel(actor_critic,device_ids=[0, 1])
        actor_critic = actor_critic_parallel.module
    if args.half_precision:
        actor_critic.half()  # convert to half precision
        for layer in actor_critic.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()    
    actor_critic.to(device, non_blocking=True)
    from pytorch_wrappers import VecPyTorch,  VecPyTorchFrameStack
    envs = VecPyTorch(envs, device, half_precision=args.half_precision)
    if args.env == 'ObtRetro-baseline':
        envs = VecPyTorchFrameStack(envs, 4, device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env.split('-')[0].lower()))

        gail_train_loader = torch.utils.data.DataLoader(
            gail.ExpertDataset(
                file_name, num_trajectories=4, subsample_frequency=20),
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, ([envs.vector_obs_len]), envs.action_space,
                              actor_critic.recurrent_hidden_state_size)
    if args.half_precision:
        rollouts.half()
    obs, vector_obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.vector_obs[0].copy_(vector_obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=100)
    episode_floors = deque(maxlen=100)
    episode_times = deque(maxlen=100)
    # history_column_names = ['AgentId', 'Start', 'Seed', 'Floor', 'Reward', 'Steps', 'Time']
    # history_column_types = {'AgentId':np.int, 'Start':np.int, 'Seed':np.int, 'Floor':np.int, 'Reward':np.float, 'Steps':np.int, 'Time':np.float}
    # try:
    #     history_df = pd.read_csv(history_file, dtype={'AgentId':np.int, 'Start': np.int,'Seed':np.int,'Floor': np.int,'Steps':np.int},)
    # except FileNotFoundError:
    #     history_df = pd.DataFrame(columns = history_column_names).astype( dtype=history_column_types)
    #     history_df.to_csv(history_file, encoding='utf-8', index=False)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], 
                    rollouts.vector_obs[step],
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # action_cpu = action.cpu() # send a copy to the cpu

            # Obser reward and next obs
            obs, vector_obs, reward, done, infos = envs.step(action)

            # for i in range(len(action)):
            #     info = infos[i]
            #     # actual_action = action if 'actual_action' not in info.keys() else info['actual_action']
            #     # action[i][0]=int(actual_action)
            #     if 'actual_action' in info.keys() and int(info['actual_action']) != int(action_cpu[i][0]):
            #         action[i][0]=int(info['actual_action'])

            history_is_dirty = False
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    episode_floors.append(int(info['episode']['floor']))
                    episode_times.append(info['episode']['l'])
            #         data = [int(info['episode']['agent']), 
            #                 int(info['episode']['start']), int(info['episode']['seed']), int(info['episode']['floor']),
            #                 np.around(info['episode']['r'],6), int(info['episode']['l']), info['episode']['t']]
            #         new_line = pd.DataFrame([data], columns = history_column_names).astype( dtype=history_column_types)
            #         history_df = new_line.append(history_df) 
            #         history_is_dirty = True
            # if history_is_dirty:
            #     history_df.to_csv(history_file, encoding='utf-8', index=False)

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
                # [[0.0] if done_ else [1.0] for done_ in done]).to(device)
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
                #  for info in infos]).to(device)
            if args.half_precision:
                masks = masks.half()
                bad_masks = bad_masks.half()
            rollouts.insert(obs, vector_obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], 
                rollouts.vector_obs[-1],
                rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], 
                    rollouts.vector_obs[step],
                    rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            try:
                os.makedirs(save_path)
            except OSError:
                pass
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print("Save at update {} / timestep {}".format(j, total_num_steps))
            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env + ".pt"))

        if j % args.log_interval == 0:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            if len(episode_rewards) == 0:
                print(
                    "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}"
                    .format(j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            0,0,# len(episode_rewards), np.mean(episode_rewards),
                            0,0,# np.median(episode_rewards), np.min(episode_rewards),
                            0,# np.max(episode_rewards), 
                            dist_entropy, value_loss,
                            action_loss))
            else:
                writer.add_scalar('reward', np.average(episode_rewards), global_step=total_num_steps)
                writer.add_scalar('floor', np.average(episode_floors), global_step=total_num_steps)
                writer.add_scalar('reward.std', np.std(episode_rewards), global_step=total_num_steps)
                writer.add_scalar('floor.std', np.std(episode_floors), global_step=total_num_steps)
                writer.add_scalar('steps', np.average(episode_times), global_step=total_num_steps)
                # writer.add_scalar('median', np.median(episode_rewards), global_step=total_num_steps)
                # writer.add_scalar('min', np.min(episode_rewards), global_step=total_num_steps)
                # writer.add_scalar('max', np.max(episode_rewards), global_step=total_num_steps)
                writer.add_scalar('FPS', int(total_num_steps / (end - start)), global_step=total_num_steps)
                writer.add_scalar('value_loss', np.around(value_loss,6), global_step=total_num_steps)
                writer.add_scalar("action_loss:", np.around(action_loss,6), global_step=total_num_steps)
                writer.add_scalar("dist_entropy:", np.around(dist_entropy,6), global_step=total_num_steps)
                print(
                    "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}"
                    .format(j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards), dist_entropy, value_loss,
                            action_loss))
            print ("value_loss:", np.around(value_loss,6), "action_loss:", np.around(action_loss,6) , "dist_entropy:", np.around(dist_entropy,6))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env, args.seed,
                     args.num_processes, eval_log_dir, device)

    # print ("step_profile:")
    # # print (step_profile)
    # #  Self CPU total %   Self CPU total      CPU total %        CPU total     CPU time avg     CUDA total %       CUDA total    CUDA time avg  Number of Calls
    # print (step_profile.key_averages().table(sort_by="self_cpu_time_total"))
    # print (step_profile.key_averages().table(sort_by="cpu_time_total"))
    # print (step_profile.key_averages().table(sort_by="cuda_time_total"))
    # print (step_profile.key_averages().table(sort_by="cpu_time"))
    # print (step_profile.key_averages().table(sort_by="cuda_time"))
    # print ("train_profile:")
    # # print (train_profile)
    # print (train_profile.key_averages().table(sort_by="self_cpu_time_total"))
    # print (train_profile.key_averages().table(sort_by="cpu_time_total"))
    # print (train_profile.key_averages().table(sort_by="cuda_time_total"))
    # print (train_profile.key_averages().table(sort_by="cpu_time"))
    # print (train_profile.key_averages().table(sort_by="cuda_time"))

if __name__ == "__main__":
    main()
