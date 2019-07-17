import os
from sohojoe_wrappers import *
from wrappers import MontezumaInfoWrapper, make_mario_env, make_robo_pong, make_robo_hockey, \
    make_multi_pong, AddRandomStateToInfo, MaxAndSkipEnv, ProcessFrame84, ExtraTimeLimit
import numpy as np
from baselines import logger
from baselines.bench import Monitor
from baselines.common.atari_wrappers import NoopResetEnv, FrameStack
import os.path as osp
from sohojoe_wrappers import done_grading, is_grading


def _make_obs_env(rank, add_monitor, args, sleep_multiple=2):
    from time import sleep
    cudaDevice = int(os.environ['CUDA_VISIBLE_DEVICES'])
    sleep_time = rank
    if cudaDevice is not -1:
        envs_per_process = args["envs_per_process"]
        # sleep_time = rank + (cudaDevice)
        sleep_time = rank + (cudaDevice * envs_per_process)
        rank = rank + (cudaDevice * envs_per_process)
    is_grading = os.getenv('OTC_EVALUATION_ENABLED', False)
    if is_grading:
        rank = 0
        sleep_time = 0
    sleep(sleep_multiple * sleep_time)
    if rank >= 35:
        rank += 1
    from sohojoe_obstacle_tower_env import ObstacleTowerEnv
    from otc_preprocessing import OTCPreprocessing
    environment_path = args['environment_filename']
    assert environment_path is not None
    real_time = args["real_time"]
    show_obs = rank == 1 or rank == 0 or args['score']
    if is_grading:
        show_obs = False
        # if args['score']:
    #     real_time = True
    if args["env"] == 'ObtRetro-v0':
        env = ObstacleTowerEnv(
            environment_path, 
            worker_id=rank, 
            timeout_wait=6000,
            retro = True,
            docker_training=args['docker_training'], 
            realtime_mode=real_time
            )
        env = OTCPreprocessing(env)
        env = ProcessFrame84Color(env)
        if show_obs:
            env = RenderObservations(env)
        env = FrameStack(env, 4)
    elif args["env"] == 'Obt-v0':
        env = ObstacleTowerEnv(
            environment_path, 
            worker_id=rank, 
            timeout_wait=6000,
            retro = False, 
            docker_training=args['docker_training'], 
            realtime_mode=real_time
            )
        env = OTCPreprocessing(env)
        env = ProcessFrame168Color(env)
        if show_obs:
            env = RenderObservations(env)
        env = FrameStack(env, 3)
    elif args["env"] == 'Obt-v1':
        env = ObstacleTowerEnv(
            environment_path, 
            worker_id=rank, 
            timeout_wait=6000,
            retro = False, 
            docker_training=args['docker_training'], 
            realtime_mode=real_time
            )
        if not is_grading:
            if args['score']:
                env = ScoreLevelWrapper(env)
            else:
                env = TrainLevelWrapper(env)
        env = OTCPreprocessing(env)
        env = ProcessFrame168Color(env)
        if show_obs:
            env = RenderObservations(env)
        env = FrameStack(env, 3)
    elif args["env"] == 'ObtRetro-v1':
        env = ObstacleTowerEnv(
            environment_path, 
            worker_id=rank, 
            timeout_wait=6000,
            retro = True, 
            docker_training=args['docker_training'], 
            realtime_mode=real_time
            )
        if not is_grading:
            if args['score']:
                env = ScoreLevelWrapper(env)
            else:
                env = TrainLevelWrapper(env)
        env = OTCPreprocessing(env)
        env = ProcessFrame84Color(env)
        if show_obs:
            env = RenderObservations(env)
        env = FrameStack(env, 3)
    elif args["env"] == 'ObtRetro-v2':
        env = ObstacleTowerEnv(
            environment_path, 
            worker_id=rank, 
            timeout_wait=6000,
            retro = True, 
            docker_training=args['docker_training'], 
            realtime_mode=real_time
            )
        if not is_grading:
            if args['score']:
                env = ScoreLevelWrapper(env)
            else:
                env = TrainLevelWrapper(env)
        env = OTCPreprocessing(env)
        env = ProcessFrame84Color(env)
        env = ColorRandomization(env)
        if show_obs:
            env = RenderObservations(env)
        env = FrameStack(env, 3)
    elif args["env"] == 'ObtRetro-v3':
        env = ObstacleTowerEnv(
            environment_path, 
            worker_id=rank, 
            timeout_wait=6000,
            retro = True, 
            docker_training=args['docker_training'], 
            realtime_mode=real_time
            )
        # env = RetroWrapper(env) # WIP
        if not is_grading:
            if args['score']:
                env = ScoreLevelWrapper(env)
            else:
                env = TrainLevelWrapper(env)
        env = OTCPreprocessing(env)
        env = ProcessFrame84Color(env)
        if show_obs:
            env = RenderObservations(env)
        env = FrameStackMono(env, 2)
    return env 

def make_env_all_params(rank, add_monitor, args, sleep_multiple=2):
    if args["env_kind"] == 'ObstacleTowerEnv':
        env = _make_obs_env(rank, add_monitor, args, sleep_multiple)
    elif args["env_kind"] == 'atari':
        env = gym.make(args['env'])
        assert 'NoFrameskip' in env.spec.id
        env = NoopResetEnv(env, noop_max=args['noop_max'])
        env = MaxAndSkipEnv(env, skip=4)
        env = ProcessFrame84(env, crop=False)
        env = FrameStack(env, 4)
        env = ExtraTimeLimit(env, args['max_episode_steps'])
        if 'Montezuma' in args['env']:
            env = MontezumaInfoWrapper(env)
        env = AddRandomStateToInfo(env)
        if rank == 2:
            env = RenderWrapper(env)
    elif args["env_kind"] == 'mario':
        env = make_mario_env()
    elif args["env_kind"] == "retro_multi":
        env = make_multi_pong()
    elif args["env_kind"] == 'robopong':
        if args["env"] == "pong":
            env = make_robo_pong()
        elif args["env"] == "hockey":
            env = make_robo_hockey()

    if add_monitor:
        logdir = osp.join('summaries',args["exp_name"])
        logger.configure(logdir)
        env = Monitor(env, osp.join(logger.get_dir(), '%.2i' % rank))
    return env