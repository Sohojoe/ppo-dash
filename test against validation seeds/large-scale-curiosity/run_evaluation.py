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
from utils import random_agent_ob_mean_std, guess_available_gpus
import gym
import tensorflow as tf
from baselines import logger
from baselines.bench import Monitor
from sohojoe_wrappers import done_grading, is_grading
import cv2
cv2.ocl.setUseOpenCL(False)


def start_score(**args):
    available_gpus = guess_available_gpus()
    print ('-- ')
    print ('------ available_gpus -------', available_gpus)
    print ('------ selecting gpu --------', 0)
    print ('-- ')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(available_gpus[0])


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
        self.ob_mean, self.ob_std, self.ob_space, self.ac_space = random_agent_ob_mean_std(None, hps['env'], nsteps=1, load=True)
        self.env = self.make_env(258)

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
        self.agents = [
            # self.create_agent('presub095', hps),
            self.create_agent('presub089', hps),
            # self.create_agent('presub088', hps),
            # self.create_agent('presub087', hps),
            # self.create_agent('presub047', hps),
            # self.create_agent('presub018', hps),
            # self.create_agent('presub001', hps),
            # self.create_agent('presub002', hps),
            # self.create_agent('presub004', hps),
            # self.create_agent('presub005', hps),
            # self.create_agent('presub015', hps),
            # self.create_agent('presub016', hps),
            # self.create_agent('presub017', hps),
            # self.create_agent('presub019', hps),
            # self.create_agent('presub020', hps),
            # self.create_agent('presub021', hps),
        ]



    def create_agent(self, exp_name, hps):
        # graph = tf.Graph() 
        # graph.as_default()
        agent = PpoOptimizer(
            scope=exp_name,
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
            exp_name=exp_name,
        )

        agent.to_report['aux'] = tf.reduce_mean(self.feature_extractor.loss)
        agent.total_loss += agent.to_report['aux']
        agent.to_report['dyn_loss'] = tf.reduce_mean(self.dynamics.loss)
        agent.total_loss += agent.to_report['dyn_loss']
        agent.to_report['feat_var'] = tf.reduce_mean(tf.nn.moments(self.feature_extractor.features, [0, 1])[1])  

        # agent.graph = graph
        # tf.reset_default_graph()          
    
        return agent

    def score(self):
        episode_reward = 0
        episode_rewards = []
        total_episodes = 0
        samples = 0
        obs = np.empty((len(self.agents)+samples, 1, *self.ob_space.shape), np.float32)
        obs[0] = self.env.reset()
        max_level = 0
        max_levels = []
        for agent in self.agents:
            agent.no_mpi_start_interaction([self.env], nlump=self.hps['nlumps'], dynamics=self.dynamics)

        # if is_grading(self.env):
        #     while not done_grading(self.env):
        #         # run_episode(env)
        #         done = False
        #         episode_reward = 0.0
                
        #         while not done:
        #             action = env.action_space.sample()
        #             obs, reward, done, info = env.step(action)
        #             episode_reward += reward
                
        #         self.env.reset()
        #     return

        while True:
            # aa = obs.reshape([len(obs) * 1, 1, *self.ob_space.shape])

            for i in range(len(self.agents)-1):
                obs[1+i] = obs[0]
            for i in range(samples):
                mu, sigma = 0, 0.1
                noise = np.random.normal(mu, sigma, obs[0].shape) 
                obs[len(self.agents)+i] = obs[0] + noise
            # obs[1] = np.copy(obs[0])
            # obs[1] = cv2.randn(obs[1],(128),(9))
            action_scores, acs, vpreds, nlps = self.policy.inference_get_ac_value_nlp(obs)
            max_actions = np.unravel_index(action_scores.argmax(), action_scores.shape)
            max_action = max_actions[1]
            max_v = vpreds.argmax()
            max_npl = nlps.argmax()
            min_npl = nlps.argmin()
            action = acs[0] # default
            # action = int(max_action) # based on highest scoring action
            # action = int(acs[max_v]) # based on highest value
            # action = int(acs[min_npl]) # based on min npl
            # action = action_scores[min_npl].argmax()
            ob, reward, done, _ = self.env.step(action)
            obs[0] = ob
            episode_reward += reward
            if reward == 1:
                max_level += 1
            if done:
                episode_rewards.append(episode_reward)
                ave_reward = sum(episode_rewards) / len(episode_rewards)
                total_episodes +=1
                max_levels.append(max_level)
                ave_level = sum(max_levels) / len(max_levels)
                print ('ep:', total_episodes, 'level:', max_level, 'ave_level:', round(ave_level,2), 'episode_reward:', episode_reward, 'ave_reward', round(ave_reward,2))
                episode_reward = 0
                max_level = 0
                if is_grading(self.env):
                    if done_grading(self.env):
                        break
                elif total_episodes >= 25:
                    break
                obs[0] = self.env.reset()
        self.env.close()


def get_experiment_environment(**args):
    from utils import setup_tensorflow_session
    from baselines.common import set_global_seeds
    from gym.utils.seeding import hash_seed
    process_seed = args["seed"] + 1000 * 0
    process_seed = hash_seed(process_seed, max_bytes=4)
    set_global_seeds(process_seed)

    logger_context = logger.scoped_configure(dir=None,
                                             format_strs=['stdout', 'log',
                                                          'csv'])
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
    logger.configure(logdir)

    start_score(**args.__dict__)
