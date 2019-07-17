import functools
import os.path as osp
from functools import partial
import os
import gym
import tensorflow as tf
import numpy as np
from sohojoe_make_env import make_env_all_params
from cppo_agent import PpoOptimizer
from auxiliary_tasks import FeatureExtractor, InverseDynamics, VAE, JustPixels
from cnn_policy import CnnPolicy
from cppo_agent import PpoOptimizer
from dynamics import Dynamics, UNet
from utils import random_agent_ob_mean_std
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
import gym
import tensorflow as tf
from baselines import logger
from baselines.bench import Monitor


def start_score(**args):
    from utils import setup_mpi_gpus
    setup_mpi_gpus()
    make_env = partial(make_env_all_params, add_monitor=True, args=args, sleep_multiple=0)

    scorer = Scorer(make_env=make_env,
                      num_timesteps=args['num_timesteps'], hps=args,
                      envs_per_process=args['envs_per_process'])
    log, tf_sess = get_experiment_environment(**args)
    with log, tf_sess:
        logdir = logger.get_dir()
        scorer.score()

class Scorer(object):
    def __init__(self, make_env, hps, num_timesteps, envs_per_process):
        self.make_env = make_env
        self.hps = hps
        self.envs_per_process = envs_per_process
        self.num_timesteps = num_timesteps
        # self._set_env_vars()

        self.ob_mean, self.ob_std, self.ob_space, self.ac_space = random_agent_ob_mean_std(None, hps['env'], nsteps=1, load=True)
        # env = self.make_env(256, add_monitor=False, sleep_multiple=1./32)
        # self.ob_space, self.ac_space = env.observation_space, env.action_space
        # env.close()
        # del env

        self.envs = [functools.partial(self.make_env, i+256+1) for i in range(envs_per_process)]

        self.policy = CnnPolicy(
            scope='pol',
            ob_space=self.ob_space,
            ac_space=self.ac_space,
            hidsize=512,
            feat_dim=512,
            ob_mean=self.ob_mean,
            ob_std=self.ob_std,
            layernormalize=False,
            nl=tf.nn.leaky_relu)

        self.feature_extractor = {"none": FeatureExtractor,
                                  "idf": InverseDynamics,
                                  "vaesph": partial(VAE, spherical_obs=True),
                                  "vaenonsph": partial(VAE, spherical_obs=False),
                                  "pix2pix": JustPixels}[hps['feat_learning']]
        self.feature_extractor = self.feature_extractor(policy=self.policy,
                                                        features_shared_with_policy=False,
                                                        feat_dim=512,
                                                        layernormalize=hps['layernorm'])

        self.dynamics = Dynamics if hps['feat_learning'] != 'pix2pix' else UNet
        self.dynamics = self.dynamics(auxiliary_task=self.feature_extractor,
                                      predict_from_pixels=hps['dyn_from_pixels'],
                                      feat_dim=512)

        self.agent = PpoOptimizer(
            scope='ppo',
            ob_space=self.ob_space,
            ac_space=self.ac_space,
            stochpol=self.policy,
            use_news=hps['use_news'],
            gamma=hps['gamma'],
            lam=hps["lambda"],
            nepochs=hps['nepochs'],
            nminibatches=hps['nminibatches'],
            lr=hps['lr'],
            cliprange=0.1,
            nsteps_per_seg=hps['nsteps_per_seg'],
            nsegs_per_env=hps['nsegs_per_env'],
            ent_coef=hps['ent_coeff'],
            normrew=hps['norm_rew'],
            normadv=hps['norm_adv'],
            ext_coeff=hps['ext_coeff'],
            int_coeff=hps['int_coeff'],
            dynamics=self.dynamics,
            load=hps['load'],
            exp_name=hps['exp_name'],
        )

        self.agent.to_report['aux'] = tf.reduce_mean(self.feature_extractor.loss)
        self.agent.total_loss += self.agent.to_report['aux']
        self.agent.to_report['dyn_loss'] = tf.reduce_mean(self.dynamics.loss)
        self.agent.total_loss += self.agent.to_report['dyn_loss']
        self.agent.to_report['feat_var'] = tf.reduce_mean(tf.nn.moments(self.feature_extractor.features, [0, 1])[1])

    def score(self):
        self.agent.start_interaction(self.envs, nlump=self.hps['nlumps'], dynamics=self.dynamics)
        from time import sleep
        sleep(2)
        episode_reward = 0
        episode_rewards = []
        total_episodes = 0
        max_level = 0
        max_levels = []
        while True:
            # info = self.agent.step()
            # self.agent.rollout.collect_rollout()
            obs, prevrews, news, infos = self.agent.rollout.env_get(0)
            if prevrews is not None:
                episode_reward += prevrews
                if prevrews == 1:
                    max_level += 1
                if news:
                    episode_rewards.append(episode_reward)
                    ave_reward = sum(episode_rewards) / len(episode_rewards)
                    total_episodes +=1
                    max_levels.append(max_level)
                    ave_level = sum(max_levels) / len(max_levels)
                    ave_level = np.around(ave_level, 2)
                    ave_reward = np.around(ave_reward, 2)
                    print ('ep:', total_episodes, 'level:', max_level, 'ave_level:', ave_level, 'episode_reward:', episode_reward, 'ave_reward', ave_reward)
                    episode_reward = 0
                    max_level = 0
                    if total_episodes >= 25:
                        break
            # acs, vpreds, nlps = self.agent.rollout.policy.get_ac_value_nlp(obs)
            # self.agent.rollout.env_step(0, acs)
            acs, vpreds, nlps = self.policy.get_ac_value_nlp(obs)
            self.agent.rollout.env_step(0, acs)
            self.agent.rollout.step_count += 1

        self.agent.stop_interaction()


def get_experiment_environment(**args):
    from utils import setup_mpi_gpus, setup_tensorflow_session
    from baselines.common import set_global_seeds
    from gym.utils.seeding import hash_seed
    process_seed = args["seed"] + 1000 * MPI.COMM_WORLD.Get_rank()
    process_seed = hash_seed(process_seed, max_bytes=4)
    set_global_seeds(process_seed)
    # setup_mpi_gpus()

    logger_context = logger.scoped_configure(dir=None,
                                             format_strs=['stdout', 'log',
                                                          'csv'] if MPI.COMM_WORLD.Get_rank() == 0 else ['log'])
    tf_context = setup_tensorflow_session()
    return logger_context, tf_context


def add_environments_params(parser):
    # parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4',type=str)
    parser.add_argument('--max-episode-steps', help='maximum number of timesteps for episode', default=4500, type=int)
    # parser.add_argument('--env_kind', type=str, default="atari")
    parser.add_argument('--env_kind', type=str, default="ObstacleTowerEnv")
    # parser.add_argument('--env', help='environment ID', default='ObtRetro-v0',type=str)
    # parser.add_argument('--env', help='environment ID', default='Obt-v0',type=str)
    parser.add_argument('--env', help='environment ID', default='ObtRetro-v3',type=str)
    parser.add_argument('--noop_max', type=int, default=30)


def add_optimization_params(parser):
    parser.add_argument('--lambda', type=float, default=0.95)
    parser.add_argument('--gamma', type=float, default=0.96)
    parser.add_argument('--nminibatches', type=int, default=8)
    parser.add_argument('--norm_adv', type=int, default=1)
    parser.add_argument('--norm_rew', type=int, default=0)
    # parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=0)
    parser.add_argument('--ent_coeff', type=float, default=0.001)
    parser.add_argument('--nepochs', type=int, default=8)
    # parser.add_argument('--num_timesteps', type=int, default=int(1e8))
    parser.add_argument('--num_timesteps', type=int, default=int(.25e8))


def add_rollout_params(parser):
    parser.add_argument('--nsteps_per_seg', type=int, default=512)
    parser.add_argument('--nsegs_per_env', type=int, default=1)
    # parser.add_argument('--envs_per_process', type=int, default=128)
    parser.add_argument('--envs_per_process', type=int, default=1)
    parser.add_argument('--nlumps', type=int, default=1)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_environments_params(parser)
    add_optimization_params(parser)
    add_rollout_params(parser)

    parser.add_argument('--exp_name', type=str, default='submission')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--dyn_from_pixels', type=int, default=0)
    parser.add_argument('--use_news', type=int, default=0)
    parser.add_argument('--ext_coeff', type=float, default=0.)
    parser.add_argument('--int_coeff', type=float, default=1.)
    parser.add_argument('--layernorm', type=int, default=0)
    parser.add_argument('--feat_learning', type=str, default="none",
                        choices=["none", "idf", "vaesph", "vaenonsph", "pix2pix"])
    parser.add_argument('--score', type=bool, default=True)
    parser.add_argument('--load', type=bool, default=True)
    parser.add_argument('--real_time', action='store_true')
    parser.add_argument('environment_filename', default='../ObstacleTower/obstacletower', nargs='?')
    parser.add_argument('--docker_training', action='store_true')
    parser.set_defaults(docker_training=False)

    args = parser.parse_args()
    logdir = osp.join('summaries',args.exp_name)
    os.environ['OPENAI_LOGDIR'] = logdir
    os.environ['OPENAI_LOG_FORMAT'] = 'stdout,tensorboard'

    start_score(**args.__dict__)
